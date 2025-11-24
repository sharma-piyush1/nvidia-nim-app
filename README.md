# NVIDIA NIM Application

This repository contains a Python application built to experiment with and run workflows utilizing **NVIDIA NIM**. The project includes environment-based configurations, dataset files, and multiple application entry points to support scalable development and experimentation.

---

## Project Structure

```bash
NVIDIA NIM/
│
├── app.py # Main application logic
├── app2.py # Secondary app variant or extended functionality
├── requirements.txt # Python dependencies
│
├── us_census/ # Dataset folder
│ ├── acsbr-015.pdf
│ ├── acsbr-016.pdf
│ ├── acsbr-017.pdf
│ └── p70-178.pdf
```

---

## Setup and Installation

Clone the repository:

git clone https://github.com/sharma-piyush1/nvidia-nim-app.git

cd nvidia-nim-app


Install dependencies:
```bash
pip install -r requirements.txt
```


Create a `.env` file based on your configuration needs (API keys, model IDs, etc.).

---

## Running the Application

To run the main application:

```bash
python app.py
```

---

## Requirements

All dependencies are managed through `requirements.txt`.

Ensure NVIDIA-compatible GPU and environment if required by NIM components.

---

## Planned Enhancements

- Add inference benchmarking
- Extend dataset automation
- Add UI or API endpoint

---

## License

MIT License
