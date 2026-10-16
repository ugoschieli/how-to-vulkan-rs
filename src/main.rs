use std::{f32::consts::PI, fs::File, io::BufReader, mem::offset_of, slice::from_ref};

use ash::{
    Device, Entry, Instance, khr,
    vk::{self, PipelineColorBlendAttachmentState},
};
use ash_window::{create_surface, enumerate_required_extensions};
use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec2, Vec3, Vec4};
use ktx::KtxInfo;
use log::{debug, info};
use obj::Obj;
use shader_slang::{
    CompileTarget, CompilerOptions, Downcast, GlobalSession, Session, SessionDesc, TargetDesc,
};
use vk_mem::{
    Alloc, Allocation, AllocationCreateFlags, AllocationCreateInfo, Allocator,
    AllocatorCreateFlags, AllocatorCreateInfo, MemoryUsage,
};
use winit::{
    application::ApplicationHandler,
    dpi::PhysicalSize,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    raw_window_handle::{HasDisplayHandle, HasWindowHandle, RawDisplayHandle},
    window::{Window, WindowAttributes, WindowId},
};

const MAX_FRAME_IN_FLIGHT: u32 = 2;

struct App {
    entry: Entry,
    instance: Instance,
    instance_surface_khr: khr::surface::Instance,
    physical_device: vk::PhysicalDevice,
    vertices: Vec<Vertex>,
    indices: Vec<u16>,
    textures: Vec<TextureData>,
    compiler: Session,
    engine: Option<Engine>,
}

struct Engine {
    window: Window,
    frame_index: usize,
    surface: vk::SurfaceKHR,
    device: Device,
    queue: vk::Queue,
    swapchain_loader: khr::swapchain::Device,
    swapchain: vk::SwapchainKHR,
    swapchain_images: Vec<vk::Image>,
    swapchain_images_views: Vec<vk::ImageView>,
    depth_image: Image,
    model_buffer: Buffer,
    vertex_buffer_size: u64,
    shader_data_buffers: Vec<(Buffer, u64)>,
    textures: Vec<Image>,
    fences: Vec<vk::Fence>,
    image_acquired_semaphores: Vec<vk::Semaphore>,
    render_complete_semaphores: Vec<vk::Semaphore>,
    command_pool: vk::CommandPool,
    command_buffers: Vec<vk::CommandBuffer>,
    desc_set: vk::DescriptorSet,
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
    allocator: Allocator,
}

impl Drop for Engine {
    fn drop(&mut self) {
        unsafe {
            self.device.device_wait_idle().unwrap();
            self.device.destroy_image_view(self.depth_image.view, None);
            self.allocator
                .destroy_image(self.depth_image.handle, &mut self.depth_image.alloc);

            self.allocator
                .destroy_buffer(self.model_buffer.handle, &mut self.model_buffer.alloc);

            for b in &mut self.shader_data_buffers {
                self.allocator.destroy_buffer(b.0.handle, &mut b.0.alloc);
            }

            for i in &mut self.textures {
                self.device.destroy_image_view(i.view, None);
                self.allocator.destroy_image(i.handle, &mut i.alloc);
            }
        }
    }
}

struct Image {
    handle: vk::Image,
    view: vk::ImageView,
    alloc: Allocation,
    sampler: Option<vk::Sampler>,
}

struct Buffer {
    handle: vk::Buffer,
    alloc: Allocation,
}

#[repr(C)]
#[derive(Debug, Pod, Zeroable, Clone, Copy)]
struct Vertex {
    pos: Vec3,
    normal: Vec3,
    uv: Vec2,
}

#[repr(C)]
#[derive(Debug, Pod, Zeroable, Clone, Copy)]
struct ShaderData {
    proj: Mat4,
    view: Mat4,
    model: [Mat4; 3],
    light_pos: Vec4,
    selected: u32,
    _pad: [u32; 3],
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        if self.engine.is_none() {
            let window_attributes =
                WindowAttributes::default().with_inner_size(PhysicalSize::new(1920, 1080));

            let window = event_loop.create_window(window_attributes).unwrap();

            let surface = unsafe {
                create_surface(
                    &self.entry,
                    &self.instance,
                    event_loop.display_handle().unwrap().as_raw(),
                    window.window_handle().unwrap().as_raw(),
                    None,
                )
                .unwrap()
            };
            let surface_caps = unsafe {
                self.instance_surface_khr
                    .get_physical_device_surface_capabilities(self.physical_device, surface)
                    .unwrap()
            };
            info!("{:#?}", surface_caps);

            let (device, queue, queue_family) = create_device(&self.instance, self.physical_device);

            let mut alloc_create_info =
                AllocatorCreateInfo::new(&self.instance, &device, self.physical_device);
            alloc_create_info.flags = AllocatorCreateFlags::BUFFER_DEVICE_ADDRESS;
            let allocator = unsafe { Allocator::new(alloc_create_info).unwrap() };

            let swapchain_loader = khr::swapchain::Device::new(&self.instance, &device);
            let (swapchain, swapchain_images, swapchain_images_views) = create_swapchain(
                &self.instance_surface_khr,
                self.physical_device,
                &device,
                &swapchain_loader,
                surface,
                &surface_caps,
            );

            let fences = create_fences(&device);
            let (image_acquired_semaphores, render_complete_semaphores) =
                create_semaphores(&device, &swapchain_images);
            let (command_pool, command_buffers) = create_command_buffers(&device, queue_family);

            let depth_format = get_depth_format(&self.instance, self.physical_device);
            let depth_image = create_depth_image(&device, &allocator, &surface_caps, depth_format);
            let (model_buffer, vertex_buffer_size) =
                upload_models(&allocator, &self.vertices, &self.indices);
            let shader_data_buffers = create_uniform_buffers(&device, &allocator);
            let textures =
                create_textures(&device, &allocator, queue, command_pool, &self.textures);

            let (desc_layout, desc_set) = create_descriptors(&device, &textures);

            let spirv = compile_shader(&self.compiler);
            let module = create_shader_module(&device, &spirv);

            let (pipeline, pipeline_layout) = create_pipeline(
                &device,
                desc_layout,
                module,
                vk::Format::B8G8R8A8_SRGB,
                depth_format,
            );

            let engine = Engine {
                window,
                frame_index: 0,
                surface,
                device,
                queue,
                swapchain_loader,
                swapchain,
                swapchain_images,
                swapchain_images_views,
                depth_image,
                model_buffer,
                vertex_buffer_size,
                shader_data_buffers,
                textures,
                fences,
                image_acquired_semaphores,
                render_complete_semaphores,
                command_pool,
                command_buffers,
                desc_set,
                pipeline,
                pipeline_layout,
                allocator,
            };
            engine.window.request_redraw();
            self.engine = Some(engine);
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        let e = self.engine.as_mut().unwrap();
        let device = &e.device;
        let window_size = &e.window.inner_size();

        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::RedrawRequested => {
                let fences = [e.fences[e.frame_index]];
                let mut image_index;

                unsafe {
                    device.wait_for_fences(&fences, true, u64::MAX).unwrap();
                    device.reset_fences(&fences).unwrap();
                    (image_index, _) = e
                        .swapchain_loader
                        .acquire_next_image(
                            e.swapchain,
                            u64::MAX,
                            e.image_acquired_semaphores[e.frame_index],
                            vk::Fence::null(),
                        )
                        .unwrap();
                }

                let shader_data = ShaderData {
                    proj: glam::camera::rh::proj::directx::perspective(
                        PI / 4.,
                        window_size.width as f32 / window_size.height as f32,
                        0.1,
                        32.0,
                    ),
                    view: Mat4::from_translation(Vec3::new(0.0, 0.0, -6.0)),
                    model: (0..3)
                        .map(|i| {
                            let instance_pos = Vec3::new(3.0 * (i - 1) as f32, 0.0, 0.0);
                            Mat4::from_translation(instance_pos)
                        })
                        .collect::<Vec<_>>()
                        .try_into()
                        .unwrap(),
                    light_pos: Vec4::new(0.0, -10.0, 10.0, 0.0),
                    selected: 1,
                    _pad: [0; 3],
                };

                let (buf, _) = &e.shader_data_buffers[e.frame_index];
                let mapped = e.allocator.get_allocation_info(&buf.alloc).mapped_data;
                let dst = unsafe {
                    std::slice::from_raw_parts_mut(mapped.cast::<u8>(), size_of::<ShaderData>())
                };
                dst.copy_from_slice(bytemuck::bytes_of(&shader_data));

                let cb = e.command_buffers[e.frame_index];
                unsafe {
                    device
                        .reset_command_buffer(cb, vk::CommandBufferResetFlags::RELEASE_RESOURCES)
                        .unwrap()
                };

                let begin_info = vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
                unsafe { device.begin_command_buffer(cb, &begin_info).unwrap() };

                let barriers = [
                    vk::ImageMemoryBarrier2::default()
                        .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                        .src_access_mask(vk::AccessFlags2::NONE)
                        .dst_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                        .dst_access_mask(
                            vk::AccessFlags2::COLOR_ATTACHMENT_READ
                                | vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
                        )
                        .old_layout(vk::ImageLayout::UNDEFINED)
                        .new_layout(vk::ImageLayout::ATTACHMENT_OPTIMAL)
                        .image(e.swapchain_images[image_index as usize])
                        .subresource_range(
                            vk::ImageSubresourceRange::default()
                                .aspect_mask(vk::ImageAspectFlags::COLOR)
                                .level_count(1)
                                .layer_count(1),
                        ),
                    vk::ImageMemoryBarrier2::default()
                        .src_stage_mask(vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS)
                        .src_access_mask(vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE)
                        .dst_stage_mask(vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS)
                        .dst_access_mask(vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE)
                        .old_layout(vk::ImageLayout::UNDEFINED)
                        .new_layout(vk::ImageLayout::ATTACHMENT_OPTIMAL)
                        .image(e.depth_image.handle)
                        .subresource_range(
                            vk::ImageSubresourceRange::default()
                                .aspect_mask(
                                    vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL,
                                )
                                .level_count(1)
                                .layer_count(1),
                        ),
                ];

                let barrier_deps_info =
                    vk::DependencyInfo::default().image_memory_barriers(&barriers);

                unsafe {
                    device.cmd_pipeline_barrier2(cb, &barrier_deps_info);
                }

                let color_attachments_info = [vk::RenderingAttachmentInfo::default()
                    .image_view(e.swapchain_images_views[image_index as usize])
                    .image_layout(vk::ImageLayout::ATTACHMENT_OPTIMAL)
                    .load_op(vk::AttachmentLoadOp::CLEAR)
                    .store_op(vk::AttachmentStoreOp::STORE)
                    .clear_value(vk::ClearValue {
                        color: vk::ClearColorValue {
                            float32: [0.0, 0.0, 0.2, 1.0],
                        },
                    })];

                let depth_attachment_info = vk::RenderingAttachmentInfo::default()
                    .image_view(e.depth_image.view)
                    .image_layout(vk::ImageLayout::ATTACHMENT_OPTIMAL)
                    .load_op(vk::AttachmentLoadOp::CLEAR)
                    .store_op(vk::AttachmentStoreOp::DONT_CARE)
                    .clear_value(vk::ClearValue {
                        depth_stencil: vk::ClearDepthStencilValue {
                            depth: 1.0,
                            stencil: 0,
                        },
                    });

                let rendering_info = vk::RenderingInfo::default()
                    .render_area(vk::Rect2D::default().extent(vk::Extent2D {
                        width: window_size.width,
                        height: window_size.height,
                    }))
                    .layer_count(1)
                    .color_attachments(&color_attachments_info)
                    .depth_attachment(&depth_attachment_info);

                unsafe {
                    device.cmd_begin_rendering(cb, &rendering_info);
                }

                let viewports = [vk::Viewport::default()
                    .width(window_size.width as f32)
                    .height(window_size.height as f32)
                    .min_depth(0.0)
                    .max_depth(1.0)];

                let scissors = [vk::Rect2D::default().extent(vk::Extent2D {
                    width: window_size.width,
                    height: window_size.height,
                })];

                unsafe {
                    device.cmd_set_viewport(cb, 0, &viewports);
                    device.cmd_set_scissor(cb, 0, &scissors);
                    device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::GRAPHICS, e.pipeline);
                    device.cmd_bind_descriptor_sets(
                        cb,
                        vk::PipelineBindPoint::GRAPHICS,
                        e.pipeline_layout,
                        0,
                        &[e.desc_set],
                        &[],
                    );
                    device.cmd_bind_vertex_buffers(
                        cb,
                        0,
                        &[e.model_buffer.handle],
                        &[vk::DeviceSize::from(0u32)],
                    );
                    device.cmd_bind_index_buffer(
                        cb,
                        e.model_buffer.handle,
                        vk::DeviceSize::from(e.vertex_buffer_size),
                        vk::IndexType::UINT16,
                    );
                    device.cmd_push_constants(
                        cb,
                        e.pipeline_layout,
                        vk::ShaderStageFlags::VERTEX,
                        0,
                        bytemuck::bytes_of(&e.shader_data_buffers[e.frame_index].1),
                    );
                    device.cmd_draw_indexed(cb, self.indices.len() as u32, 3, 0, 0, 0);
                    device.cmd_end_rendering(cb);
                }

                let barrier_present = vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                    .src_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
                    .dst_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                    .dst_access_mask(vk::AccessFlags2::NONE)
                    .old_layout(vk::ImageLayout::ATTACHMENT_OPTIMAL)
                    .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
                    .image(e.swapchain_images[image_index as usize])
                    .subresource_range(
                        vk::ImageSubresourceRange::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .level_count(1)
                            .layer_count(1),
                    );

                let barrier_present_dep_info =
                    vk::DependencyInfo::default().image_memory_barriers(from_ref(&barrier_present));

                unsafe {
                    device.cmd_pipeline_barrier2(cb, &barrier_present_dep_info);
                    device.end_command_buffer(cb).unwrap();
                }

                let wait_semaphore_info = vk::SemaphoreSubmitInfo::default()
                    .semaphore(e.image_acquired_semaphores[e.frame_index])
                    .stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT);

                let signal_semaphore_info = vk::SemaphoreSubmitInfo::default()
                    .semaphore(e.render_complete_semaphores[image_index as usize]);

                let cb_submit_info = vk::CommandBufferSubmitInfo::default().command_buffer(cb);

                let submit_info = vk::SubmitInfo2::default()
                    .wait_semaphore_infos(from_ref(&wait_semaphore_info))
                    .command_buffer_infos(from_ref(&cb_submit_info))
                    .signal_semaphore_infos(from_ref(&signal_semaphore_info));

                unsafe {
                    device
                        .queue_submit2(e.queue, from_ref(&submit_info), e.fences[e.frame_index])
                        .unwrap()
                };

                e.frame_index = ((e.frame_index as u32 + 1) % MAX_FRAME_IN_FLIGHT) as usize;

                let present_info = vk::PresentInfoKHR::default()
                    .wait_semaphores(from_ref(
                        &e.render_complete_semaphores[image_index as usize],
                    ))
                    .swapchains(from_ref(&e.swapchain))
                    .image_indices(from_ref(&image_index));

                unsafe {
                    e.swapchain_loader
                        .queue_present(e.queue, &present_info)
                        .unwrap()
                };

                e.window.request_redraw();
            }
            _ => {}
        }
    }
}

fn main() {
    pretty_env_logger::formatted_builder()
        .filter_level(log::LevelFilter::Info)
        .parse_default_env()
        .init();

    let event_loop = EventLoop::new().unwrap();
    let display_handle = event_loop.display_handle().unwrap().as_raw();

    let entry = unsafe { Entry::load().unwrap() };
    let instance = create_vulkan_instance(display_handle, &entry);
    let instance_surface_khr = khr::surface::Instance::new(&entry, &instance);
    let physical_device = get_physical_device(&instance);

    let mut obj = Obj::load("./assets/suzanne.obj").expect("Failed to load model file");
    obj.load_mtls().expect("Failed to load model materials");

    let data = &obj.data;
    let mut vertices: Vec<Vertex> = vec![];
    let mut indices: Vec<u16> = vec![];
    // Load vertex and index data
    for poly in data.objects[0].groups[0].polys.iter() {
        // tinyobjloader triangulates by default, so fan-triangulate any quads/n-gons here
        for i in 1..poly.0.len() - 1 {
            for &obj::IndexTuple(p, t, n) in [&poly.0[0], &poly.0[i], &poly.0[i + 1]] {
                let pos = data.position[p];
                let normal = data.normal[n.expect("Model is missing normals")];
                let uv = data.texture[t.expect("Model is missing texture coordinates")];
                vertices.push(Vertex {
                    pos: Vec3::new(pos[0], -pos[1], pos[2]),
                    normal: Vec3::new(normal[0], -normal[1], normal[2]),
                    uv: Vec2::new(uv[0], 1.0 - uv[1]),
                });
                indices.push(indices.len() as u16);
            }
        }
    }
    let textures = load_textures();

    let compiler = init_slang();

    let mut app = App {
        entry,
        instance,
        instance_surface_khr,
        physical_device,
        vertices,
        indices,
        textures,
        compiler,
        engine: None,
    };

    event_loop.set_control_flow(ControlFlow::Poll);
    event_loop.run_app(&mut app).unwrap();
}

fn init_slang() -> Session {
    let global_session = GlobalSession::new().unwrap();

    let targets = [TargetDesc::default()
        .format(CompileTarget::Spirv)
        .profile(global_session.find_profile("spirv_1_4"))];

    let options = CompilerOptions::default()
        .emit_spirv_directly(true)
        .matrix_layout_column(true);

    let session_desc = SessionDesc::default().targets(&targets).options(&options);
    let session = global_session.create_session(&session_desc).unwrap();

    session
}

fn compile_shader(session: &Session) -> Vec<u32> {
    let module = session.load_module("assets/shader.slang").unwrap();

    // Compose the module with all of its entry points (vertex + fragment)
    let mut components = vec![module.downcast().clone()];
    components.extend(module.entry_points().map(|ep| ep.downcast().clone()));

    let program = session
        .create_composite_component_type(&components)
        .unwrap();
    let linked_program = program.link().unwrap();

    // One SPIR-V module containing every entry point, for target 0
    let spirv = linked_program.target_code(0).unwrap();
    ash::util::read_spv(&mut std::io::Cursor::new(spirv.as_slice())).unwrap()
}

fn create_vulkan_instance(display_handle: RawDisplayHandle, entry: &Entry) -> Instance {
    let extensions = enumerate_required_extensions(display_handle).unwrap();

    let application_info = vk::ApplicationInfo::default()
        .application_name(c"How to Vulkan")
        .api_version(vk::make_api_version(0, 1, 3, 0));

    let create_info = vk::InstanceCreateInfo::default()
        .application_info(&application_info)
        .enabled_extension_names(extensions);

    unsafe { entry.create_instance(&create_info, None).unwrap() }
}

fn get_physical_device(instance: &Instance) -> vk::PhysicalDevice {
    let devices = unsafe { instance.enumerate_physical_devices().unwrap() };
    info!("{} available devices: ", devices.len());
    assert!(!devices.is_empty(), "no Vulkan physical device found");

    // Prefer a discrete GPU, but fall back to the first device (e.g. Apple Silicon is integrated)
    let mut selected_device = devices[0];
    let mut selected_device_index = 0;

    let mut props = Vec::new();

    for (i, d) in devices.into_iter().enumerate() {
        let mut p = vk::PhysicalDeviceProperties2::default();
        unsafe { instance.get_physical_device_properties2(d, &mut p) };
        props.push(p.properties);

        let device_type = p.properties.device_type;
        if device_type == vk::PhysicalDeviceType::DISCRETE_GPU {
            selected_device = d;
            selected_device_index = i;
        }
    }

    for (i, p) in props.iter().enumerate() {
        let device_name = p.device_name_as_c_str().unwrap().to_string_lossy();
        let device_type = p.device_type;
        if selected_device_index == i {
            info!("-> {device_name}: {device_type:?} (Selected)");
        } else {
            info!("-> {device_name}: {device_type:?}");
        }
        debug!("{:#?}", p);
    }

    selected_device
}

fn create_device(
    instance: &Instance,
    physical_device: vk::PhysicalDevice,
) -> (Device, vk::Queue, u32) {
    let queue_family_number =
        unsafe { instance.get_physical_device_queue_family_properties2_len(physical_device) };
    let mut queue_family_properties =
        vec![vk::QueueFamilyProperties2::default(); queue_family_number];
    unsafe {
        instance.get_physical_device_queue_family_properties2(
            physical_device,
            &mut queue_family_properties,
        )
    };

    let mut selected_queue_family = 0u32;
    for (i, q) in queue_family_properties.iter().enumerate() {
        debug!("{q:?}");
        if q.queue_family_properties
            .queue_flags
            .contains(vk::QueueFlags::GRAPHICS)
        {
            selected_queue_family = i as u32;
        }
    }

    let queue_create_infos = [vk::DeviceQueueCreateInfo::default()
        .queue_family_index(selected_queue_family)
        .queue_priorities(&[1.0])];

    let enabled_device_extensions = [khr::swapchain::NAME.as_ptr()];
    let enabled_features_10 = vk::PhysicalDeviceFeatures::default().sampler_anisotropy(true);
    let mut enabled_features_12 = vk::PhysicalDeviceVulkan12Features::default()
        .descriptor_indexing(true)
        .shader_sampled_image_array_non_uniform_indexing(true)
        .descriptor_binding_variable_descriptor_count(true)
        .runtime_descriptor_array(true)
        .buffer_device_address(true);
    let mut enabled_features_13 = vk::PhysicalDeviceVulkan13Features::default()
        .dynamic_rendering(true)
        .synchronization2(true);

    let device_create_info = vk::DeviceCreateInfo::default()
        .queue_create_infos(&queue_create_infos)
        .enabled_extension_names(&enabled_device_extensions)
        .enabled_features(&enabled_features_10)
        .push_next(&mut enabled_features_12)
        .push_next(&mut enabled_features_13);

    let device = unsafe {
        instance
            .create_device(physical_device, &device_create_info, None)
            .unwrap()
    };

    let queue = unsafe { device.get_device_queue(selected_queue_family, 0) };

    (device, queue, selected_queue_family)
}

fn create_swapchain(
    instance: &khr::surface::Instance,
    physical_device: vk::PhysicalDevice,
    device: &Device,
    swapchain_loader: &khr::swapchain::Device,
    surface: vk::SurfaceKHR,
    surface_caps: &vk::SurfaceCapabilitiesKHR,
) -> (vk::SwapchainKHR, Vec<vk::Image>, Vec<vk::ImageView>) {
    let supported_formats = unsafe {
        instance
            .get_physical_device_surface_formats(physical_device, surface)
            .unwrap()
    };
    info!("{:#?}", supported_formats);

    let supported_present_modes = unsafe {
        instance
            .get_physical_device_surface_present_modes(physical_device, surface)
            .unwrap()
    };
    info!("{:#?}", supported_present_modes);

    let format = vk::Format::B8G8R8A8_SRGB;
    let create_info = vk::SwapchainCreateInfoKHR::default()
        .surface(surface)
        .present_mode(vk::PresentModeKHR::FIFO)
        .image_format(format)
        .image_color_space(vk::ColorSpaceKHR::SRGB_NONLINEAR)
        .image_extent(surface_caps.current_extent)
        .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
        .image_array_layers(1)
        .pre_transform(vk::SurfaceTransformFlagsKHR::IDENTITY)
        .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
        .min_image_count(surface_caps.min_image_count);
    let swapchain = unsafe {
        swapchain_loader
            .create_swapchain(&create_info, None)
            .unwrap()
    };

    let swapchain_images = unsafe { swapchain_loader.get_swapchain_images(swapchain).unwrap() };

    let views = swapchain_images
        .iter()
        .map(|i| {
            let view_info = vk::ImageViewCreateInfo::default()
                .image(*i)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(format)
                .subresource_range(
                    vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .level_count(1)
                        .layer_count(1),
                );

            unsafe { device.create_image_view(&view_info, None).unwrap() }
        })
        .collect::<Vec<_>>();

    (swapchain, swapchain_images, views)
}

fn get_depth_format(instance: &Instance, physical_device: vk::PhysicalDevice) -> vk::Format {
    // D24_UNORM_S8_UINT is not available everywhere (e.g. Apple GPUs), so pick the first supported one
    let candidates = [
        vk::Format::D32_SFLOAT_S8_UINT,
        vk::Format::D24_UNORM_S8_UINT,
    ];
    candidates
        .into_iter()
        .find(|&format| {
            let props =
                unsafe { instance.get_physical_device_format_properties(physical_device, format) };
            props
                .optimal_tiling_features
                .contains(vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT)
        })
        .expect("No supported depth format found")
}

fn create_depth_image(
    device: &Device,
    allocator: &Allocator,
    surface_caps: &vk::SurfaceCapabilitiesKHR,
    depth_format: vk::Format,
) -> Image {
    let image_info = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .format(depth_format)
        .extent(vk::Extent3D {
            width: surface_caps.current_extent.width,
            height: surface_caps.current_extent.height,
            depth: 1,
        })
        .mip_levels(1)
        .array_layers(1)
        .samples(vk::SampleCountFlags::TYPE_1)
        .tiling(vk::ImageTiling::OPTIMAL)
        .usage(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT)
        .initial_layout(vk::ImageLayout::UNDEFINED);

    let alloc_info = AllocationCreateInfo {
        usage: MemoryUsage::Auto,
        flags: AllocationCreateFlags::DEDICATED_MEMORY,
        ..Default::default()
    };

    let (handle, alloc) = unsafe { allocator.create_image(&image_info, &alloc_info).unwrap() };

    let view_info = vk::ImageViewCreateInfo::default()
        .image(handle)
        .view_type(vk::ImageViewType::TYPE_2D)
        .format(depth_format)
        .subresource_range(
            vk::ImageSubresourceRange::default()
                .aspect_mask(vk::ImageAspectFlags::DEPTH)
                .level_count(1)
                .layer_count(1),
        );

    let view = unsafe { device.create_image_view(&view_info, None).unwrap() };

    Image {
        handle,
        view,
        alloc,
        sampler: None,
    }
}

fn upload_models(allocator: &Allocator, vertices: &[Vertex], indices: &[u16]) -> (Buffer, u64) {
    let vbuf_size = (vertices.len() * size_of::<Vertex>()) as vk::DeviceSize;
    let ibuf_size = (indices.len() * size_of::<u16>()) as vk::DeviceSize;

    let buffer_info = vk::BufferCreateInfo::default()
        .size(vbuf_size + ibuf_size)
        .usage(vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::INDEX_BUFFER);

    let alloc_info = AllocationCreateInfo {
        usage: MemoryUsage::Auto,
        flags: AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE
            | AllocationCreateFlags::HOST_ACCESS_ALLOW_TRANSFER_INSTEAD
            | AllocationCreateFlags::MAPPED,
        ..Default::default()
    };

    let (handle, alloc) = unsafe { allocator.create_buffer(&buffer_info, &alloc_info).unwrap() };
    let mapped = allocator.get_allocation_info(&alloc).mapped_data;
    assert!(!mapped.is_null());

    let dst = unsafe {
        std::slice::from_raw_parts_mut(mapped.cast::<u8>(), (vbuf_size + ibuf_size) as usize)
    };
    let (vdst, idst) = dst.split_at_mut(vbuf_size as usize);
    vdst.copy_from_slice(bytemuck::cast_slice(vertices));
    idst.copy_from_slice(bytemuck::cast_slice(indices));

    (Buffer { handle, alloc }, vbuf_size)
}

fn create_uniform_buffers(device: &Device, allocator: &Allocator) -> Vec<(Buffer, u64)> {
    let buffers = (0..MAX_FRAME_IN_FLIGHT)
        .into_iter()
        .map(|_| {
            let buffer_info = vk::BufferCreateInfo::default()
                .size(size_of::<ShaderData>() as vk::DeviceSize)
                .usage(vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS);

            let alloc_info = AllocationCreateInfo {
                usage: MemoryUsage::Auto,
                flags: AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE
                    | AllocationCreateFlags::HOST_ACCESS_ALLOW_TRANSFER_INSTEAD
                    | AllocationCreateFlags::MAPPED,
                ..Default::default()
            };

            let (handle, alloc) =
                unsafe { allocator.create_buffer(&buffer_info, &alloc_info).unwrap() };

            let device_address_info = vk::BufferDeviceAddressInfo::default().buffer(handle);
            let device_address = unsafe { device.get_buffer_device_address(&device_address_info) };

            (Buffer { handle, alloc }, device_address)
        })
        .collect();

    buffers
}

fn create_fences(device: &Device) -> Vec<vk::Fence> {
    let fence_info = vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);

    let fences: Vec<_> = (0..MAX_FRAME_IN_FLIGHT)
        .into_iter()
        .map(|_| unsafe { device.create_fence(&fence_info, None).unwrap() })
        .collect();

    fences
}

fn create_semaphores(
    device: &Device,
    swapchain_images: &[vk::Image],
) -> (Vec<vk::Semaphore>, Vec<vk::Semaphore>) {
    let semaphore_info = vk::SemaphoreCreateInfo::default();

    let image_acquired_semaphores: Vec<_> = (0..MAX_FRAME_IN_FLIGHT)
        .into_iter()
        .map(|_| unsafe { device.create_semaphore(&semaphore_info, None).unwrap() })
        .collect();

    let render_complete_semaphores: Vec<_> = (0..swapchain_images.len())
        .into_iter()
        .map(|_| unsafe { device.create_semaphore(&semaphore_info, None).unwrap() })
        .collect();

    (image_acquired_semaphores, render_complete_semaphores)
}

fn create_command_buffers(
    device: &Device,
    queue_family: u32,
) -> (vk::CommandPool, Vec<vk::CommandBuffer>) {
    let pool_info = vk::CommandPoolCreateInfo::default()
        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
        .queue_family_index(queue_family);

    let pool = unsafe { device.create_command_pool(&pool_info, None).unwrap() };

    let buffers_info = vk::CommandBufferAllocateInfo::default()
        .command_pool(pool)
        .command_buffer_count(MAX_FRAME_IN_FLIGHT);

    let buffers = unsafe { device.allocate_command_buffers(&buffers_info).unwrap() };

    (pool, buffers)
}

struct TextureData {
    width: u32,
    height: u32,
    format: vk::Format,
    /// All mip levels packed back to back, level 0 first
    data: Vec<u8>,
    /// Byte offset of each mip level inside `data`
    mip_offsets: Vec<u64>,
}

fn load_textures() -> Vec<TextureData> {
    (0..3)
        .map(|i| {
            let filename = format!("./assets/suzanne{i}.ktx");
            let file = BufReader::new(File::open(&filename).expect("Failed to open texture file"));
            let decoder = ktx::Decoder::new(file).expect("Failed to read KTX header");

            let header = decoder.header();

            let format = match header.gl_internal_format() {
                0x8C43 => vk::Format::R8G8B8A8_SRGB,  // GL_SRGB8_ALPHA8
                0x8058 => vk::Format::R8G8B8A8_UNORM, // GL_RGBA8
                f => panic!("{filename}: unsupported glInternalFormat {f:#x}"),
            };

            let mut data = Vec::new();
            let mut mip_offsets = Vec::with_capacity(header.mipmap_levels() as usize);
            for level in decoder.read_textures() {
                mip_offsets.push(data.len() as u64);
                data.extend_from_slice(&level);
            }

            TextureData {
                width: header.pixel_width(),
                height: header.pixel_height(),
                format,
                data,
                mip_offsets,
            }
        })
        .collect()
}

fn create_textures(
    device: &Device,
    allocator: &Allocator,
    queue: vk::Queue,
    pool: vk::CommandPool,
    textures: &[TextureData],
) -> Vec<Image> {
    let images = textures
        .into_iter()
        .map(|tex| {
            let image_info = vk::ImageCreateInfo::default()
                .image_type(vk::ImageType::TYPE_2D)
                .format(tex.format)
                .extent(
                    vk::Extent3D::default()
                        .width(tex.width)
                        .height(tex.height)
                        .depth(1),
                )
                .mip_levels(tex.mip_offsets.len() as u32)
                .array_layers(1)
                .samples(vk::SampleCountFlags::TYPE_1)
                .tiling(vk::ImageTiling::OPTIMAL)
                .usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED)
                .initial_layout(vk::ImageLayout::UNDEFINED);

            let alloc_info = AllocationCreateInfo {
                usage: MemoryUsage::Auto,
                ..Default::default()
            };

            let (handle, alloc) =
                unsafe { allocator.create_image(&image_info, &alloc_info).unwrap() };

            let view_info = vk::ImageViewCreateInfo::default()
                .image(handle)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(image_info.format)
                .subresource_range(
                    vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .level_count(tex.mip_offsets.len() as u32)
                        .layer_count(1),
                );

            let view = unsafe { device.create_image_view(&view_info, None).unwrap() };

            let staging_buffer_info = vk::BufferCreateInfo::default()
                .size(tex.data.len() as vk::DeviceSize)
                .usage(vk::BufferUsageFlags::TRANSFER_SRC);

            let staging_alloc_info = AllocationCreateInfo {
                usage: MemoryUsage::Auto,
                flags: AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE
                    | AllocationCreateFlags::MAPPED,
                ..Default::default()
            };

            let (staging_buffer, mut staging_alloc) = unsafe {
                allocator
                    .create_buffer(&staging_buffer_info, &staging_alloc_info)
                    .unwrap()
            };

            let stagging_alloc_info = allocator.get_allocation_info(&staging_alloc);
            let mapped = stagging_alloc_info.mapped_data;
            let dst =
                unsafe { std::slice::from_raw_parts_mut(mapped.cast::<u8>(), tex.data.len()) };
            dst.copy_from_slice(&tex.data);

            let staging_fence_info = vk::FenceCreateInfo::default();
            let staging_fence = unsafe { device.create_fence(&staging_fence_info, None).unwrap() };

            let staging_command_buffer_info = vk::CommandBufferAllocateInfo::default()
                .command_pool(pool)
                .command_buffer_count(1);

            let staging_command_buffer = unsafe {
                device
                    .allocate_command_buffers(&staging_command_buffer_info)
                    .unwrap()[0]
            };

            let begin_info = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            unsafe {
                device
                    .begin_command_buffer(staging_command_buffer, &begin_info)
                    .unwrap()
            }

            let barrier_tex_image = vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::NONE)
                .src_access_mask(vk::AccessFlags2::NONE)
                .dst_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .image(handle)
                .subresource_range(
                    vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .level_count(tex.mip_offsets.len() as u32)
                        .layer_count(1),
                );

            let barriers = [barrier_tex_image];
            let barrier_info = vk::DependencyInfo::default().image_memory_barriers(&barriers);
            unsafe { device.cmd_pipeline_barrier2(staging_command_buffer, &barrier_info) };

            let copy_regions = tex
                .mip_offsets
                .iter()
                .enumerate()
                .map(|(i, offset)| {
                    vk::BufferImageCopy::default()
                        .buffer_offset(*offset)
                        .image_subresource(
                            vk::ImageSubresourceLayers::default()
                                .aspect_mask(vk::ImageAspectFlags::COLOR)
                                .mip_level(i as u32)
                                .layer_count(1),
                        )
                        .image_extent(
                            vk::Extent3D::default()
                                .width(tex.width >> i)
                                .height(tex.height >> i)
                                .depth(1),
                        )
                })
                .collect::<Vec<_>>();

            unsafe {
                device.cmd_copy_buffer_to_image(
                    staging_command_buffer,
                    staging_buffer,
                    handle,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &copy_regions,
                );
            }

            let barrier_tex_read = vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
                .dst_access_mask(vk::AccessFlags2::SHADER_READ)
                .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .new_layout(vk::ImageLayout::READ_ONLY_OPTIMAL)
                .image(handle)
                .subresource_range(
                    vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .level_count(tex.mip_offsets.len() as u32)
                        .layer_count(1),
                );

            let barriers = [barrier_tex_read];
            let barrier_info = vk::DependencyInfo::default().image_memory_barriers(&barriers);

            unsafe {
                device.cmd_pipeline_barrier2(staging_command_buffer, &barrier_info);
                device.end_command_buffer(staging_command_buffer).unwrap();
            };

            let binding =
                [vk::CommandBufferSubmitInfo::default().command_buffer(staging_command_buffer)];
            let submit_info = vk::SubmitInfo2::default().command_buffer_infos(&binding);

            unsafe {
                device
                    .queue_submit2(queue, &[submit_info], staging_fence)
                    .unwrap();
                device
                    .wait_for_fences(&[staging_fence], true, u64::MAX)
                    .unwrap();
                allocator.destroy_buffer(staging_buffer, &mut staging_alloc);
            };

            let sampler_info = vk::SamplerCreateInfo::default()
                .mag_filter(vk::Filter::LINEAR)
                .min_filter(vk::Filter::LINEAR)
                .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
                .anisotropy_enable(true)
                .max_anisotropy(8.0)
                .max_lod(tex.mip_offsets.len() as f32);

            let sampler = unsafe { device.create_sampler(&sampler_info, None).unwrap() };

            Image {
                alloc,
                handle,
                view,
                sampler: Some(sampler),
            }
        })
        .collect();

    images
}

fn create_descriptors(
    device: &Device,
    textures: &[Image],
) -> (vk::DescriptorSetLayout, vk::DescriptorSet) {
    let flags = [vk::DescriptorBindingFlags::VARIABLE_DESCRIPTOR_COUNT];
    let mut flags = vk::DescriptorSetLayoutBindingFlagsCreateInfo::default().binding_flags(&flags);

    let bindings = [vk::DescriptorSetLayoutBinding::default()
        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
        .stage_flags(vk::ShaderStageFlags::FRAGMENT)
        .descriptor_count(textures.len() as u32)];

    let layout_info = vk::DescriptorSetLayoutCreateInfo::default()
        .bindings(&bindings)
        .push_next(&mut flags);

    let texture_desc_set_layout = unsafe {
        device
            .create_descriptor_set_layout(&layout_info, None)
            .unwrap()
    };

    let pool_size = [vk::DescriptorPoolSize::default()
        .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
        .descriptor_count(textures.len() as u32)];
    let pool_size = vk::DescriptorPoolCreateInfo::default()
        .max_sets(1)
        .pool_sizes(&pool_size);

    let desc_pool = unsafe { device.create_descriptor_pool(&pool_size, None).unwrap() };

    let desc_count = [textures.len() as u32];
    let mut desc_alloc_info = vk::DescriptorSetVariableDescriptorCountAllocateInfo::default()
        .descriptor_counts(&desc_count);

    let layouts = [texture_desc_set_layout];
    let desc_alloc_info = vk::DescriptorSetAllocateInfo::default()
        .descriptor_pool(desc_pool)
        .set_layouts(&layouts)
        .push_next(&mut desc_alloc_info);

    let desc_set = unsafe { device.allocate_descriptor_sets(&desc_alloc_info).unwrap()[0] };

    let images_info = textures
        .iter()
        .map(|i| {
            vk::DescriptorImageInfo::default()
                .image_view(i.view)
                .sampler(i.sampler.unwrap())
                .image_layout(vk::ImageLayout::READ_ONLY_OPTIMAL)
        })
        .collect::<Vec<_>>();

    let write = [vk::WriteDescriptorSet::default()
        .dst_set(desc_set)
        .dst_binding(0)
        .descriptor_count(textures.len() as u32)
        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
        .image_info(&images_info)];

    unsafe {
        device.update_descriptor_sets(&write, &[]);
    }

    (texture_desc_set_layout, desc_set)
}

fn create_shader_module(device: &Device, spirv: &[u32]) -> vk::ShaderModule {
    let module_info = vk::ShaderModuleCreateInfo::default().code(spirv);

    let module = unsafe { device.create_shader_module(&module_info, None).unwrap() };

    module
}

fn create_pipeline(
    device: &Device,
    descriptor_set_layout: vk::DescriptorSetLayout,
    module: vk::ShaderModule,
    image_format: vk::Format,
    depth_format: vk::Format,
) -> (vk::Pipeline, vk::PipelineLayout) {
    let push_constants = [vk::PushConstantRange::default()
        .stage_flags(vk::ShaderStageFlags::VERTEX)
        .size(size_of::<vk::DeviceAddress>() as u32)];

    let layouts = [descriptor_set_layout];
    let layout_info = vk::PipelineLayoutCreateInfo::default()
        .push_constant_ranges(&push_constants)
        .set_layouts(&layouts);

    let layout = unsafe { device.create_pipeline_layout(&layout_info, None).unwrap() };

    let vertex_binding = [vk::VertexInputBindingDescription::default()
        .binding(0)
        .stride(size_of::<Vertex>() as u32)
        .input_rate(vk::VertexInputRate::VERTEX)];

    let vertex_attrs = [
        vk::VertexInputAttributeDescription::default()
            .location(0)
            .binding(0)
            .format(vk::Format::R32G32B32_SFLOAT)
            .offset(offset_of!(Vertex, pos) as u32),
        vk::VertexInputAttributeDescription::default()
            .location(1)
            .binding(0)
            .format(vk::Format::R32G32B32_SFLOAT)
            .offset(offset_of!(Vertex, normal) as u32),
        vk::VertexInputAttributeDescription::default()
            .location(2)
            .binding(0)
            .format(vk::Format::R32G32_SFLOAT)
            .offset(offset_of!(Vertex, uv) as u32),
    ];

    let vertex_input_state = vk::PipelineVertexInputStateCreateInfo::default()
        .vertex_binding_descriptions(&vertex_binding)
        .vertex_attribute_descriptions(&vertex_attrs);

    let input_assembly_state = vk::PipelineInputAssemblyStateCreateInfo::default()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST);

    let shader_stages = [
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(module)
            .name(c"main"),
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(module)
            .name(c"main"),
    ];

    let viewport_state = vk::PipelineViewportStateCreateInfo::default()
        .viewport_count(1)
        .scissor_count(1);

    let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
    let dynamic_state =
        vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

    let depth_stencil_state = vk::PipelineDepthStencilStateCreateInfo::default()
        .depth_test_enable(true)
        .depth_write_enable(true)
        .depth_compare_op(vk::CompareOp::LESS_OR_EQUAL);

    let image_format = [image_format];
    let mut rendering_info = vk::PipelineRenderingCreateInfo::default()
        .color_attachment_formats(&image_format)
        .depth_attachment_format(depth_format);

    let blend_attachments = [PipelineColorBlendAttachmentState::default()
        .color_write_mask(vk::ColorComponentFlags::RGBA)];
    let blend = vk::PipelineColorBlendStateCreateInfo::default().attachments(&blend_attachments);

    let raster = vk::PipelineRasterizationStateCreateInfo::default().line_width(1.0);

    let multisample = vk::PipelineMultisampleStateCreateInfo::default()
        .rasterization_samples(vk::SampleCountFlags::TYPE_1);

    let pipeline_info = [vk::GraphicsPipelineCreateInfo::default()
        .push_next(&mut rendering_info)
        .stages(&shader_stages)
        .vertex_input_state(&vertex_input_state)
        .input_assembly_state(&input_assembly_state)
        .viewport_state(&viewport_state)
        .rasterization_state(&raster)
        .multisample_state(&multisample)
        .depth_stencil_state(&depth_stencil_state)
        .color_blend_state(&blend)
        .dynamic_state(&dynamic_state)
        .layout(layout)];

    let pipeline = unsafe {
        device
            .create_graphics_pipelines(vk::PipelineCache::null(), &pipeline_info, None)
            .unwrap()[0]
    };

    (pipeline, layout)
}
