import os
import argparse
import torch
from flask import Flask, request, jsonify, render_template, Response, stream_with_context
from werkzeug.utils import secure_filename
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from transformers.generation.streamers import TextIteratorStreamer
import threading
import requests
import re
import html

def _strip_html(raw):
    try:
        return re.sub(r"<[^>]+>", " ", raw).replace("&nbsp;", " ").strip()
    except Exception:
        return raw

def _clean_text(raw):
    try:
        return _strip_html(html.unescape(raw)).replace("\n", " ").strip()
    except Exception:
        return raw

def _extract_ddg_results(raw):
    titles = re.findall(r'class="result__a"[^>]*>(.*?)</a>', raw, re.DOTALL)
    snippets = re.findall(r'class="result__snippet"[^>]*>(.*?)</a>', raw, re.DOTALL)
    results = []
    for idx, title in enumerate(titles[:5]):
        snippet = snippets[idx] if idx < len(snippets) else ""
        title_text = _clean_text(title)
        snippet_text = _clean_text(snippet)
        if title_text:
            results.append(f"{idx + 1}. {title_text}\n{snippet_text}")
    return results

def _extract_bing_results(raw):
    items = re.findall(r'<li class="b_algo".*?</li>', raw, re.DOTALL)
    results = []
    for idx, item in enumerate(items[:5]):
        title_match = re.search(r"<h2>.*?<a[^>]*>(.*?)</a>.*?</h2>", item, re.DOTALL)
        snippet_match = re.search(r"<p>(.*?)</p>", item, re.DOTALL)
        title_text = _clean_text(title_match.group(1)) if title_match else ""
        snippet_text = _clean_text(snippet_match.group(1)) if snippet_match else ""
        if title_text:
            results.append(f"{idx + 1}. {title_text}\n{snippet_text}")
    return results

def _web_search(query):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0 Safari/537.36"
    }
    errors = []
    weather_text, weather_error = _fetch_weather(query)
    if weather_text:
        return weather_text, None
    if weather_error:
        errors.append(weather_error)
    try:
        resp = requests.get("https://duckduckgo.com/html/", params={"q": query}, headers=headers, timeout=8)
        if resp.status_code == 200:
            results = _extract_ddg_results(resp.text)
            if results:
                return "\n\n".join(results), None
        else:
            errors.append(f"duckduckgo http {resp.status_code}")
    except Exception as e:
        errors.append(f"duckduckgo {e}")
    try:
        resp = requests.get("https://cn.bing.com/search", params={"q": query}, headers=headers, timeout=8)
        if resp.status_code == 200:
            results = _extract_bing_results(resp.text)
            if results:
                return "\n\n".join(results), None
        else:
            errors.append(f"bing http {resp.status_code}")
    except Exception as e:
        errors.append(f"bing {e}")
    return "", "; ".join(errors) if errors else "search failed"

def _extract_city_from_query(query):
    if not query:
        return None
    match = re.search(r"(.+?)(天气|气温|温度|下雨|降雨)", query)
    if not match:
        return None
    city = match.group(1)
    city = re.sub(r"(今天|现在|当前|明天|后天|本周|这周|周末|最近|一下|一下吧|一下呢|如何|怎么样|怎样)", "", city)
    return city.strip() if city.strip() else None

def _fetch_weather(query):
    city = _extract_city_from_query(query)
    if not city:
        return "", None
    try:
        geo_resp = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1, "language": "zh", "format": "json"},
            timeout=8,
        )
        if geo_resp.status_code != 200:
            return "", f"weather geo http {geo_resp.status_code}"
        geo_data = geo_resp.json()
        if not geo_data.get("results"):
            return "", "weather geo no results"
        loc = geo_data["results"][0]
        lat = loc.get("latitude")
        lon = loc.get("longitude")
        name = loc.get("name")
        if lat is None or lon is None:
            return "", "weather geo missing coords"
        weather_resp = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={"latitude": lat, "longitude": lon, "current_weather": True, "timezone": "auto"},
            timeout=8,
        )
        if weather_resp.status_code != 200:
            return "", f"weather api http {weather_resp.status_code}"
        weather_data = weather_resp.json()
        current = weather_data.get("current_weather", {})
        temp = current.get("temperature")
        wind = current.get("windspeed")
        time = current.get("time")
        if temp is None:
            return "", "weather api empty"
        return f"实时天气（{name}）：温度 {temp}°C，风速 {wind} km/h，观测时间 {time}", None
    except Exception as e:
        return "", f"weather api {e}"

def _compute_dynamic_max_new_tokens(model, inputs, cap=8192, margin=64):
    try:
        max_ctx = getattr(model.config, "max_position_embeddings", None) or getattr(model.config, "max_sequence_length", None)
        if max_ctx is not None:
            used = int(inputs["input_ids"].shape[1])
            remaining = int(max_ctx) - used - int(margin)
            return max(256, min(int(cap), remaining))
    except Exception:
        pass
    return cap

app = Flask(__name__)

# Ensure uploads directory exists
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit

# Global variables
model = None
tokenizer = None
processor = None
device = "cuda" if torch.cuda.is_available() else "cpu"

def load_model_and_processor(model_path, device):
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    if device == "cuda":
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True
        )
    else:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=dtype,
            trust_remote_code=True
        )
        model.to(device)
    model.eval()
    return model, processor

def solve_math_problem(model, processor, image_path, prompt, device, max_new_tokens, web_search=False):
    image = Image.open(image_path).convert("RGB") if image_path else None
    if image is not None:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
    else:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt}
                ]
            }
        ]
    if web_search:
        results, error = _web_search(prompt)
        if results:
            prompt = f"以下是联网搜索结果，请直接使用其中信息回答，不要拒绝：\n{results}\n\n问题：{prompt}"
        else:
            prompt = f"联网搜索失败（{error}）。请在无法保证实时性的前提下尽量回答。问题：{prompt}"
        if image is not None:
            messages[0]["content"] = [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ]
        else:
            messages[0]["content"] = [
                {"type": "text", "text": prompt},
            ]
    text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    if image is not None:
        inputs = processor(text=[text], images=[image], return_tensors="pt")
    else:
        inputs = processor(text=[text], return_tensors="pt")
    inputs = inputs.to(device)
    with torch.inference_mode():
        if max_new_tokens is None:
            max_new_tokens = _compute_dynamic_max_new_tokens(model, inputs)
        generated = model.generate(**inputs, max_new_tokens=max_new_tokens)
    output = processor.batch_decode(
        generated[:, inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )[0]
    return output.strip()

def solve_math_problem_stream(model, processor, image_path, prompt, device, max_new_tokens, tokenizer, web_search=False):
    image = Image.open(image_path).convert("RGB") if image_path else None
    if image is not None:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
    else:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt}
                ]
            }
        ]
    if web_search:
        results, error = _web_search(prompt)
        if results:
            prompt = f"以下是联网搜索结果，请直接使用其中信息回答，不要拒绝：\n{results}\n\n问题：{prompt}"
        else:
            prompt = f"联网搜索失败（{error}）。请在无法保证实时性的前提下尽量回答。问题：{prompt}"
        if image is not None:
            messages[0]["content"] = [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ]
        else:
            messages[0]["content"] = [
                {"type": "text", "text": prompt},
            ]
    text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    if image is not None:
        inputs = processor(text=[text], images=[image], return_tensors="pt")
    else:
        inputs = processor(text=[text], return_tensors="pt")
    inputs = inputs.to(device)
    streamer = TextIteratorStreamer(
        tokenizer or processor.tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
    )
    gen_kwargs = dict(**inputs)
    if max_new_tokens is not None:
        gen_kwargs["max_new_tokens"] = max_new_tokens
    else:
        gen_kwargs["max_new_tokens"] = _compute_dynamic_max_new_tokens(model, inputs)
    t = threading.Thread(target=model.generate, kwargs={**gen_kwargs, "streamer": streamer})
    t.start()
    for new_text in streamer:
        yield new_text
    t.join()

def init_model(model_path):
    global model, tokenizer, processor
    print(f"Initializing model from {model_path} on {device}...")
    try:
        model, processor = load_model_and_processor(model_path, device=device)
        tokenizer = getattr(processor, "tokenizer", None)
        print("Model initialized successfully.")
    except Exception as e:
        print(f"Failed to initialize model: {e}")
        # We don't exit here to allow the UI to load even if model fails (for debugging UI), 
        # but inference will fail.
        pass

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    global model, tokenizer, processor
    
    if not model:
        return jsonify({'error': '模型未加载，请检查服务器日志。'}), 500

    file = request.files.get('file') if 'file' in request.files else None
    prompt = request.form.get('prompt')
    if not prompt or prompt.strip() == "":
        prompt = '请一步一步详细推理这张图片中的数学题，并给出最终答案。'
    if file is None and (prompt is None or prompt.strip() == ""):
        return jsonify({'error': '没有上传文件或文本'}), 400
    if file is not None and file.filename == '':
        return jsonify({'error': '未选择文件'}), 400

    deep_think_value = request.form.get('deep_think', '0')
    deep_think = deep_think_value in ['1', 'true', 'True', 'yes', 'on']
    web_search_value = request.form.get('web_search', '0')
    web_search = web_search_value in ['1', 'true', 'True', 'yes', 'on']
    max_new_tokens = None if deep_think else 1024

    filepath = None
    filename = None
    if file is not None:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
    try:
        if filename:
            print(f"Processing request for {filename} with prompt: {prompt}")
        else:
            print(f"Processing request with prompt: {prompt}")
        result = solve_math_problem(
            model,
            processor,
            filepath,
            prompt,
            device=device,
            max_new_tokens=max_new_tokens,
            web_search=web_search,
        )
        return jsonify({'result': result})
    except Exception as e:
        print(f"Inference error: {e}")
        return jsonify({'error': str(e)}), 500
    finally:
        try:
            if filepath and os.path.exists(filepath):
                os.remove(filepath)
        except:
            pass

@app.route('/predict_stream', methods=['POST'])
def predict_stream():
    global model, tokenizer, processor
    if not model:
        return jsonify({'error': '模型未加载，请检查服务器日志。'}), 500
    file = request.files.get('file') if 'file' in request.files else None
    prompt = request.form.get('prompt')
    if not prompt or prompt.strip() == "":
        prompt = '请一步一步详细推理这张图片中的数学题，并给出最终答案。'
    deep_think_value = request.form.get('deep_think', '0')
    deep_think = deep_think_value in ['1', 'true', 'True', 'yes', 'on']
    web_search_value = request.form.get('web_search', '0')
    web_search = web_search_value in ['1', 'true', 'True', 'yes', 'on']
    max_new_tokens = None if deep_think else 1024

    if file is None and (prompt is None or prompt.strip() == ""):
        return jsonify({'error': '没有上传文件或文本'}), 400
    if file is not None and file.filename == '':
        return jsonify({'error': '未选择文件'}), 400

    filepath = None
    if file is not None:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

    def generate():
        try:
            for chunk in solve_math_problem_stream(
                model,
                processor,
                filepath,
                prompt,
                device,
                max_new_tokens,
                tokenizer,
                web_search=web_search,
            ):
                yield chunk
        except Exception as e:
            yield f"\n[错误] {str(e)}"
        finally:
            try:
                if filepath and os.path.exists(filepath):
                    os.remove(filepath)
            except:
                pass

    return Response(stream_with_context(generate()), mimetype='text/plain; charset=utf-8')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Math Model WebUI")
    # Default path from user's provided command
    parser.add_argument("--model_path", type=str, default=r"E:\套瓷三剑客\Nano-Math-plus\merged_safetensors\global_step_650", help="Path to the merged model")
    parser.add_argument("--port", type=int, default=6006, help="Port to run the web server on")
    
    args, unknown = parser.parse_known_args()
    
    # Check if default path exists, if not, try local relative path for Windows users
    if not os.path.exists(args.model_path):
        # Assuming we are in E:\套瓷三剑客\math\WebUI
        # And model is in E:\套瓷三剑客\math\merged_safetensors\global_step_650
        local_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../merged_safetensors/global_step_650'))
        if os.path.exists(local_path):
            print(f"Warning: Path {args.model_path} not found. Switching to local path: {local_path}")
            args.model_path = local_path
        else:
            print(f"Warning: Model path {args.model_path} does not exist. Inference will fail unless corrected.")

    init_model(args.model_path)
    app.run(host='0.0.0.0', port=args.port, debug=False)
