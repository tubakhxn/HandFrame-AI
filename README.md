# HandFrame AI 

An interactive computer vision project that lets users create a floating frame with their hands, capture their face using a pinch gesture, and instantly transform it into multiple AI-generated artistic styles.

Built for experimentation with real-time computer vision, gesture recognition, and AI-powered image generation.

---

## Creator

**Developer:** Tuba Khan (@tubakhxn)

GitHub:
https://github.com/tubakhxn

Project:
https://github.com/tubakhxn/HandFrame-AI

---

## Features

- Real-time hand tracking
- Gesture-controlled floating frame
- Pinch-to-capture interaction
- 16+ AI artistic styles
- Perspective-aware floating image panel
- Real-time webcam rendering
- Smooth landmark filtering
- Asynchronous AI inference
- OpenCV fallback stylization
- Modular project architecture

---

## Tech Stack

- Python
- OpenCV
- MediaPipe
- NumPy
- PyTorch
- FLUX.1
- Computer Vision

---

## Project Structure

| File | Description |
|------|-------------|
| hand_tracker.py | Hand tracking & gesture recognition |
| smoothing.py | Landmark smoothing |
| perspective.py | Perspective transforms & image warping |
| styles.py | AI style presets |
| inference.py | FLUX inference engine |
| ui_overlay.py | HUD & interface |
| app.py | Main application |

---

## Installation

```bash
git clone https://github.com/tubakhxn/HandFrame-AI.git

cd HandFrame-AI

pip install -r requirements.txt

python app.py
```

To use a local FLUX model:

```bash
python app.py --model_path YOUR_MODEL_PATH
```

---

## Controls

- L-Hand Frame → Open floating panel
- Pinch Gesture → Capture image
- `[` / `]` → Change style
- `R` → Reset
- `Q` → Quit

---

## Contributing

Contributions are welcome.

1. Fork the repository
2. Create a new feature branch
3. Commit your changes
4. Push your branch
5. Open a Pull Request

---

## License

MIT License

---

 If you found this project useful, consider starring the repository.