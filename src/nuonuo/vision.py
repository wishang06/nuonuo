import logging
import threading
import time
import urllib.request
from pathlib import Path
from typing import Callable

import cv2

from nuonuo.clock import SessionClock
from nuonuo.perception import Perception, PerceptionEvent
from nuonuo.pose import head_pose_from_matrix
from nuonuo.state import SharedState
from nuonuo.timeline import BehaviourTimeline

log = logging.getLogger("nuonuo.vision")

MODELS = {
    "face_landmarker.task": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
    "gesture_recognizer.task": "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task",
}


def ensure_models(models_dir: Path) -> dict[str, Path]:
    models_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, url in MODELS.items():
        path = models_dir / name
        if not path.exists() or path.stat().st_size == 0:
            log.info("Downloading %s ...", name)
            tmp = path.with_suffix(".part")
            urllib.request.urlretrieve(url, tmp)
            tmp.replace(path)
        paths[name] = path
    return paths


class VisionWorker(threading.Thread):
    def __init__(
        self,
        state: SharedState,
        on_event: Callable[[PerceptionEvent], None],
        models_dir: Path,
        user_name: str = "Andy",
        camera_index: int = 0,
        timeline: BehaviourTimeline | None = None,
        clock: SessionClock | None = None,
    ):
        super().__init__(name="vision", daemon=True)
        self.state = state
        self.on_event = on_event
        self.models_dir = models_dir
        self.camera_index = camera_index
        self.timeline = timeline
        self.clock = clock or SessionClock()
        self.perception = Perception(user_name=user_name)

    def run(self) -> None:
        try:
            self._run()
        except Exception:
            log.exception("Vision worker crashed; continuing without vision")
            self.state.set(status="vision error")
        finally:
            cv2.destroyAllWindows()

    def _run(self) -> None:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

        paths = ensure_models(self.models_dir)
        mode = mp_vision.RunningMode.VIDEO
        face = mp_vision.FaceLandmarker.create_from_options(
            mp_vision.FaceLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=str(paths["face_landmarker.task"])),
                running_mode=mode,
                num_faces=1,
                output_face_blendshapes=True,
                output_facial_transformation_matrixes=True,
            )
        )
        gest = mp_vision.GestureRecognizer.create_from_options(
            mp_vision.GestureRecognizerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=str(paths["gesture_recognizer.task"])),
                running_mode=mode,
                num_hands=1,
            )
        )
        cap = None
        t0 = time.monotonic()
        last_ts = -1
        preview_open = False
        try:
            while not self.state.stop.is_set():
                if not self.state.vision_enabled:
                    if cap is not None:
                        cap.release()
                        cap = None
                    if preview_open:
                        cv2.destroyAllWindows()
                        preview_open = False
                    time.sleep(0.2)
                    continue
                if cap is None:
                    cap = self._open_camera()
                ok, frame = cap.read()
                if not ok:
                    time.sleep(0.05)
                    continue
                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                now = time.monotonic()
                ts = max(int((now - t0) * 1000), last_ts + 1)
                last_ts = ts
                fr = face.detect_for_video(image, ts)
                gr = gest.recognize_for_video(image, ts)

                blend = yaw = pitch = None
                if fr.face_blendshapes:
                    blend = {c.category_name: c.score for c in fr.face_blendshapes[0]}
                    if fr.facial_transformation_matrixes:
                        pose = head_pose_from_matrix(fr.facial_transformation_matrixes[0])
                        if pose is not None:
                            yaw, pitch, _roll = pose
                gesture = wrist_x = None
                if gr.gestures and gr.gestures[0]:
                    top = gr.gestures[0][0]
                    if top.score > 0.5 and top.category_name != "None":
                        gesture = top.category_name
                if gr.hand_landmarks:
                    wrist_x = gr.hand_landmarks[0][0].x

                ev = self.perception.update(now, blend, yaw, gesture, wrist_x, pitch)
                if self.timeline is not None:
                    self.timeline.observe(self.clock.t_ms(now), self.perception.state_label())
                self.state.set(mood=self.perception.mood)
                if ev:
                    self.state.set(last_event=ev.text)
                    self.on_event(ev)

                if self.state.preview:
                    self._draw(frame, fr, gr, gesture)
                    cv2.imshow("Nuonuo preview (P to hide)", frame)
                    preview_open = True
                    if cv2.waitKey(1) & 0xFF in (ord("p"), ord("P")):
                        self.state.set(preview=False)
                elif preview_open:
                    cv2.destroyAllWindows()
                    preview_open = False
        finally:
            if cap is not None:
                cap.release()
            face.close()
            gest.close()

    def _open_camera(self):
        cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW) if hasattr(cv2, "CAP_DSHOW") else None
        if cap is None or not cap.isOpened():
            cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            raise RuntimeError("Could not open webcam")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        return cap

    def _draw(self, frame, fr, gr, gesture) -> None:
        h, w = frame.shape[:2]
        for lm in fr.face_landmarks[:1]:
            for p in lm[::4]:
                cv2.circle(frame, (int(p.x * w), int(p.y * h)), 1, (0, 255, 0), -1)
        for hand in gr.hand_landmarks:
            for p in hand:
                cv2.circle(frame, (int(p.x * w), int(p.y * h)), 3, (255, 128, 0), -1)
        p = self.perception
        if p.yaw_smooth is None:
            pose_line = f"mood={p.mood} (no face)"
        elif p.pitch_smooth is None:
            pose_line = f"mood={p.mood} yaw={p.yaw_smooth:.0f}"
        else:
            pose_line = f"mood={p.mood} yaw={p.yaw_smooth:.0f} pitch={p.pitch_smooth:.0f}"
        lines = [
            pose_line,
            f"smile={p._pair('mouthSmile'):.2f} blink={p._pair('eyeBlink'):.2f} jaw={p.s('jawOpen'):.2f}",
            f"gesture={gesture}",
            " ".join(k for k, d in p.states.items() if d.value),
        ]
        for i, line in enumerate(lines):
            cv2.putText(frame, line, (8, 20 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
