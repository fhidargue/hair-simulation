from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Set

import numpy as np
from ncca.ngl import Vec3
from PySide6.QtCore import Qt

if TYPE_CHECKING:
    from PySide6.QtGui import QKeyEvent, QMouseEvent

    from ui.hair_scene import HairScene


@dataclass
class HairInputState:
    keys_pressed: Set[int]
    rotate_head: bool = False
    last_mouse_position: Optional[object] = None
    last_face_rotation: Optional[Vec3] = None


class HairInputController:
    """
    Owns keyboard/mouse state and applies input effects to the HairScene.
    Keeps HairScene smaller by isolating event-handling logic.
    """

    def __init__(self):
        self.state = HairInputState(keys_pressed=set())
        self.rotation_sensitivity = 0.3

    def key_press(self, scene: "HairScene", event: "QKeyEvent") -> None:
        """
        Handles key press events to update interaction state and trigger camera or scene controls.

        Args:
            scene: The active HairScene receiving the input.
            event: Qt key event containing the pressed key information.
        """
        self.state.keys_pressed.add(event.key())

        poly_mode = None
        match event.key():
            case Qt.Key.Key_Escape:
                scene.window().close()
            case Qt.Key.Key_W:
                poly_mode = scene._POLY_LINE
            case Qt.Key.Key_S:
                poly_mode = scene._POLY_FILL
            case Qt.Key.Key_A:
                scene.animate = not scene.animate
            case Qt.Key.Key_1:
                scene.hair_system.update(scene._DELTA_TIME)
                scene._ribbons_dirty = True

        if poly_mode is not None:
            scene.set_polygon_mode(poly_mode)

        scene.update()

    def key_release(self, scene: "HairScene", event: "QKeyEvent") -> None:
        """
        Handles key release events to update interaction state and stop active camera or scene controls.

        Args:
            scene: The active HairScene receiving the input.
            event: Qt key event containing the released key information.
        """
        self.state.keys_pressed.discard(event.key())
        scene.update()

    def mouse_press(self, scene: "HairScene", event: "QMouseEvent") -> None:
        """
        Handles mouse press events to initiate interactions such as head rotation or camera control.

        Args:
            scene: The active HairScene receiving the input.
            event: Qt mouse event containing button and position information.
        """
        scene.setFocus(Qt.FocusReason.MouseFocusReason)

        if event.button() == Qt.MouseButton.LeftButton:
            self.state.rotate_head = True
            self.state.last_mouse_position = event.position()
            self.state.last_face_rotation = Vec3(scene.face_rotation.x, scene.face_rotation.y, scene.face_rotation.z)

    def mouse_release(self, scene: "HairScene", event: "QMouseEvent") -> None:
        """
        Handles mouse release events to stop active mouse-driven interactions.

        Args:
            scene: The active HairScene receiving the input.
            event: Qt mouse event indicating which button was released.
        """
        if event.button() == Qt.MouseButton.LeftButton:
            self.state.rotate_head = False
            self.state.last_mouse_position = None
            self.state.last_face_rotation = None


    def mouse_move(self, scene: "HairScene", event: "QMouseEvent") -> None:
        """
        Handles mouse movement to rotate or manipulate the scene while a mouse action is active.

        Args:
            scene: The active HairScene being interacted with.
            event: Qt mouse event containing the current cursor position.
        """
        if not self.state.rotate_head or self.state.last_mouse_position is None:
            return

        pos = event.position()
        dx = pos.x() - self.state.last_mouse_position.x()
        dy = pos.y() - self.state.last_mouse_position.y()

        s = self.rotation_sensitivity

        # Rotate head (face_rotation is Vec3 of Euler angles)
        scene.face_rotation.y += dx * s
        scene.face_rotation.x += dy * s

        # Clamp X axis
        scene.face_rotation.x = max(-80.0, min(80.0, scene.face_rotation.x))

        # Sync hair system
        scene.hair_system.head_rotation = scene.face_rotation
        scene.sync_head_transform_now()

        # Push root positions to GPU so the frame matches immediately
        scene.line_renderer.update_buffers(scene.hair_system)
        if scene.ribbon_renderer.thickness_scale > 0.05:
            scene.ribbon_renderer.update_segments_gpu(scene.hair_system)

        scene._ribbons_dirty = True
        self.state.last_mouse_position = pos
        scene.update()


    def process_camera_movement(self, scene: "HairScene") -> None:
        """
        Updates the camera position and orientation based on the current input state (keyboard and mouse).

        Args:
            scene: The active HairScene whose camera should be updated.
        """
        x_dir = 0.0
        y_dir = 0.0

        for key in self.state.keys_pressed:
            match key:
                case Qt.Key.Key_Left:
                    y_dir = -1.0
                case Qt.Key.Key_Right:
                    y_dir = 1.0
                case Qt.Key.Key_Up:
                    x_dir = 1.0
                case Qt.Key.Key_Down:
                    x_dir = -1.0

        if x_dir or y_dir:
            scene.camera.move(x_dir, y_dir, scene._DELTA_TIME + 0.05)
            scene._camera_dirty = True

    def get_camera_pos_np(self, scene) -> np.ndarray:
        """
        Returns camera position as a numpy array.
        Supports multiple camera attribute naming conventions.
        """
        cam = scene.camera

        for attr in ("position", "pos", "_pos", "_position", "eye"):
            if hasattr(cam, attr):
                p = getattr(cam, attr)
                return np.array([p.x, p.y, p.z], dtype=np.float32)

        return np.array([0.0, 0.0, 3.0], dtype=np.float32)
