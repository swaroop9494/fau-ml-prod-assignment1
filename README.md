# Albumy

#Swaroop

*Capture and share every wonderful moment.*

> Example application for *[Python Web Development with Flask](https://helloflask.com/en/book/1)* (《[Flask Web 开发实战](https://helloflask.com/book/1)》).

Demo: http://albumy.helloflask.com

![Screenshot](https://helloflask.com/screenshots/albumy.png)

## Installation

clone:
```
$ git clone https://github.com/greyli/albumy.git
$ cd albumy
```
create & activate virtual env then install dependency:

with venv/virtualenv + pip:
```
$ python -m venv env  # use `virtualenv env` for Python2, use `python3 ...` for Python3 on Linux & macOS
$ source env/bin/activate  # use `env\Scripts\activate` on Windows
$ pip install -r requirements.txt
```
or with Pipenv:
```
$ pipenv install --dev
$ pipenv shell
```
generate fake data then run:
```
$ flask forge
$ flask run
* Running on http://127.0.0.1:5000/
```
Test account:
* email: `admin@helloflask.com`
* password: `helloflask`

## Option A — Google AI Studio (Gemini) API

This project can be extended to use **Google AI Studio (Gemini)** for automatic alt-text generation.

### Prerequisites
- Python 3.10+
- Flask dependencies (from requirements.txt)
- Pillow
- Flask-Whooshee
- `google-generativeai` package

### Install
```bash
pip install google-generativeai
```

### Setup API Key
1. Go to [Google AI Studio](https://ai.google.dev/).
2. Create a project and API key.
3. Add the key to your environment variables or `.env` file:

```
GEMINI_API_KEY=your_api_key_here
```

### Usage in Code
In your `ml_service.py` or equivalent file:
```python
import os, google.generativeai as genai

genai.configure(api_key=os.environ["GEMINI_API_KEY"])

def generate_alt_text(image_path: str) -> str:
    model = genai.GenerativeModel("gemini-1.5-flash")
    with open(image_path, "rb") as f:
        img_bytes = f.read()
    resp = model.generate_content(["Provide descriptive alt text.", {"mime_type":"image/jpeg","data":img_bytes}])
    return resp.text.strip()
```

### Example Run
- Upload an image to Albumy → alt text auto-generates using Gemini.
- Edit or regenerate alt text from the photo page.

## License

This project is licensed under the MIT License (see the
[LICENSE](LICENSE) file for details).
