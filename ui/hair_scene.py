import OpenGL.GL as gl
import numpy as np

from ncca.ngl import FirstPersonCamera, Primitives, ShaderLib, Transform, Vec3
from PySide6.QtCore import Slot
from PySide6.QtOpenGLWidgets import QOpenGLWidget

from lib.constants import (
    BACKGROUND_COLOR,
    DELTA_TIME,
    FACE_COLOR,
    FACE_MESH_PATH,
    FACE_SCALE,
    FACE_SHADER,
    FOLLOWER_SPREAD,
    HAIR_COLOR,
    HAIR_MASK_PATH,
    HAIR_MASK_THRESHOLD,
    HAIR_RIBBON_SHADER,
    HAIR_SHADER,
    HAIR_SPACING,
    HAIR_STYLES,
    HEAD_RADIUS,
    INITIAL_HAIR_LENGTH,
    INITIAL_HAIR_PARTICLE_COUNT,
    INITIAL_HAIR_STRAND_COUNT,
    INITIAL_HAIR_THICKNESS,
    MAX_ROOT_RADIUS,
    ORIGIN_POSITION,
)
from lib.hair_factory import create_head_hair_from_mesh
from lib.utils import make_strand, ngl_mat4_to_np, apply_face_transform
from module.face_mesh import FaceMesh
from module.hair_system import HairSystem
from render.hair_line_renderer import HairLineRenderer
from render.hair_ribbon_renderer import HairRibbonRenderer
from ui.hair_input import HairInputController


class HairScene(QOpenGLWidget):
    def __init__(self):
        super().__init__()
        self.ratio = self.devicePixelRatio()
        self.animate = True
        self.keys_pressed = set()
        self.rotate = False
        self.original_x_position = 0
        self.original_y_position = 0
        self._ribbons_dirty = True
        self._camera_dirty = True
        self._ribbon_frame = 0
        self._ribbon_update_every = 1

        # Input controls
        self.input = HairInputController()
        self._POLY_LINE = gl.GL_LINE
        self._POLY_FILL = gl.GL_FILL
        self._DELTA_TIME = DELTA_TIME

        # Render instances
        self.line_renderer = HairLineRenderer()
        self.ribbon_renderer = HairRibbonRenderer()
        self.ribbon_renderer.thickness_scale = INITIAL_HAIR_THICKNESS

        # Qt Signals
        self.hair_strand_count = INITIAL_HAIR_STRAND_COUNT
        self.hair_color = HAIR_COLOR
        self.hair_style = HAIR_STYLES[0]
        self.hair_length = INITIAL_HAIR_LENGTH

        self.head_position = ORIGIN_POSITION
        self.head_radius = HEAD_RADIUS
        self.face_scale = FACE_SCALE
        self.face_rotation = Vec3(0.0, 20.0, 0.0)

    @Slot(int)
    def update_hair_strand_count(self, value):
        self.hair_strand_count = value
        self._rebuild_hair()

    @Slot(Vec3)
    def update_hair_color(self, value):
        self.hair_color = value

    @Slot(int)
    def update_hair_style(self, value):
        self.hair_style = value
        self._rebuild_hair()

    @Slot(int)
    def update_hair_length(self, value):
        self.hair_length = value

    @Slot(int)
    def update_hair_thickness(self, value):
        thickness = value / 100.0
        self.ribbon_renderer.thickness_scale = thickness
        self._ribbons_dirty = True

    @Slot(int)
    def update_hair_density(self, value):
        followers = max(1, int(value))
        self.ribbon_renderer.followers_per_segment = followers

    @Slot(int)
    def update_wind(self, value):
        self.hair_system.wind_enabled = value == 2

    @Slot(int)
    def update_show_ellipsoids(self, value):
        self.hair_system.collision_debug = value == 2

    def initializeGL(self):
        """
        Initializes the OpenGL state, loads shaders, sets up the camera, and builds
        all GPU resources required for rendering the face and hair system.
        """
        gl.glClearColor(*BACKGROUND_COLOR)

        ShaderLib.load_shader(
            HAIR_SHADER, "shaders/HairVert.glsl", "shaders/HairFrag.glsl"
        )
        ShaderLib.use(HAIR_SHADER)

        ShaderLib.load_shader(
            FACE_SHADER,
            "shaders/FaceVert.glsl",
            "shaders/FaceFrag.glsl",
        )

        ShaderLib.load_shader(
            HAIR_RIBBON_SHADER,
            "shaders/HairRibbonVert.glsl",
            "shaders/HairRibbonFrag.glsl",
        )

        Primitives.load_default_primitives()

        # Camera setup
        self.camera = FirstPersonCamera(
            Vec3(0.0, 0.5, 3.0), Vec3(0.0, 0.0, 1.0), Vec3(0.0, 1.0, 0.0), 45.0
        )

        # Face mesh setup
        self.head_radius = HEAD_RADIUS
        self.head_position = ORIGIN_POSITION
        self.face = FaceMesh(FACE_MESH_PATH)
        self.face_scale = FACE_SCALE
        self.face_rotation = Vec3(0.0, 20.0, 0.0)
        self.head_rotation = ORIGIN_POSITION
        self._prev_head_matrix = None

        self.hair_system = HairSystem(
            head_position=self.head_position, head_radius=self.head_radius
        )
        self.hair_system.head_rotation = self.face_rotation

        # Draw ellipsoid collider
        self.hair_system.set_collider_from_face_mesh(
            self.face,
            shrink=0.95,
            center_y_offset=2.0,
            center_z_offset=-0.80,
            radius_x_scale=1.05,
            radius_y_scale=0.85,
            radius_z_scale=1.15,
        )

        try:
            create_head_hair_from_mesh(
                hair_system=self.hair_system,
                face_mesh=self.face,
                face_position=self.head_position,
                face_scale=self.face_scale,
                face_rotation=self.face_rotation,
                strands=self.hair_strand_count,
                spacing=HAIR_SPACING,
                particles=max(5, INITIAL_HAIR_PARTICLE_COUNT),
                make_strand_func=make_strand,
                mask_path=HAIR_MASK_PATH,
                mask_threshold=HAIR_MASK_THRESHOLD,
            )
        except Exception:
            import traceback

            traceback.print_exc()

        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_MULTISAMPLE)
        gl.glEnable(gl.GL_SAMPLE_ALPHA_TO_COVERAGE)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        gl.glDisable(gl.GL_CULL_FACE)

        self.line_renderer.create_buffers(self.hair_system)
        self.ribbon_renderer.create_buffers(self.hair_system)

        # Follower settings
        self.ribbon_renderer.follower_spread = FOLLOWER_SPREAD
        self.ribbon_renderer.max_root_radius = MAX_ROOT_RADIUS

        self.startTimer(16)

    def resizeGL(self, w: int, h: int):
        """
        Updates the camera projection when the OpenGL viewport is resized.

        Args:
            w: New viewport width in pixels.
            h: New viewport height in pixels.
        """
        if h == 0:
            return

        self.camera.set_projection(
            45.0, (w * self.ratio) / (h * self.ratio), 0.05, 200.0
        )

    def paintGL(self):
        """
        Renders a single frame: clears the framebuffer, updates the camera,
        draws the face mesh, and renders the hair using either line or ribbon shaders.
        """
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glViewport(
            0, 0, int(self.width() * self.ratio), int(self.height() * self.ratio)
        )
        gl.glEnable(gl.GL_PROGRAM_POINT_SIZE)

        self.input.process_camera_movement(self)

        light_pos = (0.0, 10.0, 10.0)

        # Render the face mesh
        ShaderLib.use(FACE_SHADER)

        face_tx = Transform()
        face_tx.set_position(
            self.head_position.x,
            self.head_position.y,
            self.head_position.z,
        )
        face_tx.set_scale(self.face_scale, self.face_scale, self.face_scale)
        face_tx.set_rotation(
            self.face_rotation.x, self.face_rotation.y, self.face_rotation.z
        )

        ShaderLib.set_uniform("MVP", self.camera.get_vp() @ face_tx.get_matrix())
        ShaderLib.set_uniform("M", face_tx.get_matrix())
        ShaderLib.set_uniform("color", *FACE_COLOR)
        ShaderLib.set_uniform("light_position", *light_pos)
        self.face.draw()

        # Draw the ellopsoid
        self._draw_debug_ellipsoid(face_tx)

        if not self.ribbon_renderer.thickness_scale > 0.05:
            ShaderLib.use(HAIR_SHADER)
            self.line_renderer.render(self.camera, self.hair_color, self.hair_style)
        else:
            ShaderLib.use(HAIR_RIBBON_SHADER)
            camera_np = self.input.get_camera_pos_np(self)

            ShaderLib.set_uniform("light_position", 0.0, 10.0, 10.0)
            ShaderLib.set_uniform(
                "camera_position",
                float(camera_np[0]),
                float(camera_np[1]),
                float(camera_np[2]),
            )
            ShaderLib.set_uniform("rim_strength", 0.25)
            ShaderLib.set_uniform("spec_strength", 0.12)

            gl.glEnable(gl.GL_DEPTH_TEST)
            gl.glDepthMask(gl.GL_FALSE)

            # Use alpha instead of blending
            gl.glDisable(gl.GL_BLEND)
            gl.glEnable(gl.GL_SAMPLE_ALPHA_TO_COVERAGE)

            self.ribbon_renderer.render(self.camera, self.hair_color, camera_np)

            # Paint the lines on top if needed
            if self.hair_style == 2:
                gl.glDisable(gl.GL_SAMPLE_ALPHA_TO_COVERAGE)
                gl.glEnable(gl.GL_BLEND)
                gl.glDepthMask(gl.GL_TRUE)
                ShaderLib.use(HAIR_SHADER)
                self.line_renderer.render(self.camera, self.hair_color, self.hair_style)
                gl.glEnable(gl.GL_SAMPLE_ALPHA_TO_COVERAGE)

            # Restore state
            gl.glDisable(gl.GL_SAMPLE_ALPHA_TO_COVERAGE)
            gl.glEnable(gl.GL_BLEND)
            gl.glDepthMask(gl.GL_TRUE)

    def timerEvent(self, event):
        """
        Advances the simulation each frame by updating head transforms,
        stepping the hair physics, refreshing GPU buffers, and triggering a redraw.
        """
        if not self.animate:
            return

        # Keep hair system head rotation synced
        self.hair_system.head_rotation = self.face_rotation
        self.hair_system.head_position = self.head_position

        face_tx = Transform()
        face_tx.set_position(
            self.head_position.x, self.head_position.y, self.head_position.z
        )
        face_tx.set_scale(self.face_scale, self.face_scale, self.face_scale)
        face_tx.set_rotation(
            self.face_rotation.x, self.face_rotation.y, self.face_rotation.z
        )

        # Use same face metrix on hair ellipsoid collider
        M_np = ngl_mat4_to_np(face_tx.get_matrix())
        self.hair_system.set_head_matrices(M_np)

        self.hair_system.update(DELTA_TIME)

        self.line_renderer.update_buffers(self.hair_system)
        self._ribbon_frame += 1

        if self.ribbon_renderer.thickness_scale > 0.05:
            if self._ribbon_frame % self._ribbon_update_every == 0:
                self.ribbon_renderer.update_segments_gpu(self.hair_system)

        self.update()

    def _rebuild_hair(self):
        """
        Recreates the hair system and all GPU buffers to reflect changes in style,
        density, or length while preserving global scene state.
        """
        self.makeCurrent()
        old_system = self.hair_system

        self.hair_system = HairSystem(
            head_position=self.head_position,
            head_radius=self.head_radius,
        )

        # Restore hair system placements
        self.hair_system.head_rotation = self.face_rotation
        self.hair_system.hair_style = self.hair_style
        self.hair_system.wind_enabled = old_system.wind_enabled
        self.hair_system.wind_strength = old_system.wind_strength
        self.hair_system.wind_radius = old_system.wind_radius

        # Draw ellipsoid collider
        self.hair_system.set_collider_from_face_mesh(
            self.face,
            shrink=0.95,
            center_y_offset=2.0,
            center_z_offset=-0.80,
            radius_x_scale=1.05,
            radius_y_scale=0.85,
            radius_z_scale=1.15,
        )

        create_head_hair_from_mesh(
            hair_system=self.hair_system,
            face_mesh=self.face,
            face_position=self.head_position,
            face_scale=self.face_scale,
            face_rotation=self.face_rotation,
            strands=self.hair_strand_count,
            spacing=HAIR_SPACING,
            particles=max(5, int(self.hair_length)),
            make_strand_func=make_strand,
            mask_path=HAIR_MASK_PATH,
            mask_threshold=HAIR_MASK_THRESHOLD,
        )

        self.line_renderer.create_buffers(self.hair_system)
        self.ribbon_renderer.create_buffers(self.hair_system)

        # Follower settings
        self.ribbon_renderer.follower_spread = FOLLOWER_SPREAD
        self.ribbon_renderer.max_root_radius = MAX_ROOT_RADIUS

        self._ribbons_dirty = True
        self._camera_dirty = True

        self.doneCurrent()

    def set_polygon_mode(self, poly_mode):
        """
        Sets the OpenGL polygon rendering mode for debugging or visualization.

        Args:
            poly_mode: OpenGL polygon mode constant (e.g. gl.GL_LINE or gl.GL_FILL).
        """
        self.makeCurrent()
        try:
            gl.glPolygonMode(gl.GL_FRONT_AND_BACK, poly_mode)
        finally:
            self.doneCurrent()

    def sync_head_transform_now(self) -> None:
        """
        Update the head matrices and pins strand roots to the current head transform
        so hair visually follows head rotation without waiting for the next timerEvent tick.
        """
        face_tx = Transform()
        face_tx.set_position(self.head_position.x, self.head_position.y, self.head_position.z)
        face_tx.set_scale(self.face_scale, self.face_scale, self.face_scale)
        face_tx.set_rotation(self.face_rotation.x, self.face_rotation.y, self.face_rotation.z)

        M_np = ngl_mat4_to_np(face_tx.get_matrix())
        self.hair_system.set_head_matrices(M_np)

        # Pin roots in world space
        for strand in self.hair_system.strands:
            if hasattr(strand, "root_local"):
                root_world = apply_face_transform(
                    strand.root_local,
                    self.head_position,
                    1.0,
                    self.face_rotation,
                )
                strand.positions_np[0] = root_world
                strand.prev_positions_np[0] = root_world

    # Keyboard events
    def keyPressEvent(self, event):
        self.input.key_press(self, event)

    def keyReleaseEvent(self, event):
        self.input.key_release(self, event)

    # Mouse events
    def mousePressEvent(self, event):
        self.input.mouse_press(self, event)

    def mouseReleaseEvent(self, event):
        self.input.mouse_release(self, event)

    def mouseMoveEvent(self, event):
        self.input.mouse_move(self, event)

    # Debugging
    def _draw_debug_ellipsoid(self, face_tx: Transform):
        """
        Renders wireframe ellipsoids for the head and facial feature colliders for debugging collisions.

        Args:
            face_tx: Transform of the face mesh used to place colliders in world space.
        """
        if not getattr(self.hair_system, "collision_debug", False):
            return
        if not (
            hasattr(self.hair_system, "collider_center_local_np")
            and hasattr(self.hair_system, "collider_radii_local_np")
        ):
            return

        # Build extra colliders
        colliders = [
            (
                "head",
                self.hair_system.collider_center_local_np,
                self.hair_system.collider_radii_local_np,
                (1.0, 0.1, 0.1),
            ),
        ]

        if hasattr(self.hair_system, "nose_center_local_np") and hasattr(
            self.hair_system, "nose_radii_local_np"
        ):
            colliders.append(
                (
                    "nose",
                    self.hair_system.nose_center_local_np,
                    self.hair_system.nose_radii_local_np,
                    (1.0, 0.2, 0.2),
                )
            )

        if hasattr(self.hair_system, "mouth_center_local_np") and hasattr(
            self.hair_system, "mouth_radii_local_np"
        ):
            colliders.append(
                (
                    "mouth",
                    self.hair_system.mouth_center_local_np,
                    self.hair_system.mouth_radii_local_np,
                    (1.0, 0.4, 0.4),
                )
            )

        ShaderLib.use(FACE_SHADER)
        ShaderLib.set_uniform("light_position", 0.0, 10.0, 10.0)

        gl.glDisable(gl.GL_CULL_FACE)
        gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)
        gl.glLineWidth(1.0)

        # Draw head with depth test ON
        gl.glEnable(gl.GL_DEPTH_TEST)
        _, c, r, rgb = colliders[0]
        c = c.astype(float)
        r = r.astype(float)

        ell_tx = Transform()
        ell_tx.set_position(float(c[0]), float(c[1]), float(c[2]))
        ell_tx.set_scale(float(r[0]), float(r[1]), float(r[2]))
        M = face_tx.get_matrix() @ ell_tx.get_matrix()

        ShaderLib.set_uniform("MVP", self.camera.get_vp() @ M)
        ShaderLib.set_uniform("M", M)
        ShaderLib.set_uniform("color", float(rgb[0]), float(rgb[1]), float(rgb[2]))
        Primitives.draw("icosahedron")

        # Draw nose/mouth with depth test
        if len(colliders) > 1:
            gl.glDisable(gl.GL_DEPTH_TEST)
            for name, c, r, rgb in colliders[1:]:
                c = c.astype(float)
                r = r.astype(float)

                ell_tx = Transform()
                ell_tx.set_position(float(c[0]), float(c[1]), float(c[2]))
                ell_tx.set_scale(float(r[0]), float(r[1]), float(r[2]))
                M = face_tx.get_matrix() @ ell_tx.get_matrix()

                ShaderLib.set_uniform("MVP", self.camera.get_vp() @ M)
                ShaderLib.set_uniform("M", M)
                ShaderLib.set_uniform(
                    "color", float(rgb[0]), float(rgb[1]), float(rgb[2])
                )
                Primitives.draw("icosahedron")

            gl.glEnable(gl.GL_DEPTH_TEST)

        # Restore
        gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)
        gl.glLineWidth(1.0)