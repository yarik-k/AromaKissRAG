#!/usr/bin/env python3

import json
import os
import time
import random
from typing import List, Dict, Any
from openai import OpenAI

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError(
        "OPENAI_API_KEY environment variable is not set. "
    )
client = OpenAI(api_key=api_key)

class CandlePostProcessor:
    def __init__(self, raw_posts_file: str):
        self.raw_posts_file = raw_posts_file
        self.posts = self.load_posts()
        
    def load_posts(self) -> List[str]:
        with open(self.raw_posts_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data if isinstance(data, list) else [data]
    
    def categorize_post(self, post: str) -> str:
        post_lower = post.lower()
        
        if "интересные факты" in post_lower:
            return "факт"
        elif "коллекция" in post_lower or "ароматы" in post_lower:
            return "каталог"
        elif "новинка" in post_lower or "новые" in post_lower:
            return "новинка"
        elif "поздрав" in post_lower or "марта" in post_lower:
            return "поздравление"
        elif "цена" in post_lower or "стоимост" in post_lower:
            return "цены"
        elif "декор" in post_lower or "цвет" in post_lower:
            return "дизайн"
        elif "воск" in post_lower or "материал" in post_lower:
            return "материалы"
        elif "готов" in post_lower and "заказ" in post_lower:
            return "готовые_заказы"
        elif "вдохновение" in post_lower or "настроение" in post_lower:
            return "вдохновение"
        else:
            return "общий"
    
    def generate_prompts(self, category: str) -> List[str]:
        prompts = {
            "факт": [
                "Напиши интересный факт о свечах",
                "Поделись познавательной информацией о свечах",
                "Расскажи что-то увлекательное про историю свечей"
            ],
            "каталог": [
                "Представь коллекцию ароматов для свечей",
                "Опиши ассортимент парфюмированных отдушек",
                "Покажи список доступных ароматов"
            ],
            "новинка": [
                "Расскажи о новых ароматах в коллекции",
                "Представь новинки для свечей",
                "Анонсируй новые продукты"
            ],
            "поздравление": [
                "Напиши поздравительный пост с праздником",
                "Создай праздничное сообщение для клиентов",
                "Поздравь подписчиков с праздником"
            ],
            "цены": [
                "Объясни ценообразование на свечи",
                "Расскажи о стоимости продукции",
                "Обоснуй цены на натуральные свечи"
            ],
            "дизайн": [
                "Опиши варианты декора для свечей",
                "Расскажи о цветовых решениях",
                "Представь дизайнерские возможности"
            ],
            "материалы": [
                "Расскажи о качестве используемых материалов",
                "Объясни преимущества кокосового воска",
                "Опиши состав натуральных свечей"
            ],
            "готовые_заказы": [
                "Покажи готовые работы",
                "Представь выполненные заказы",
                "Поделись фотографиями готовых свечей"
            ],
            "вдохновение": [
                "Создай вдохновляющий пост о свечах",
                "Поделись настроением дня",
                "Напиши атмосферный пост"
            ],
            "общий": [
                "Напиши пост о свечах ручной работы",
                "Создай сообщение для канала о свечах",
                "Расскажи о преимуществах натуральных свечей"
            ]
        }
        return prompts.get(category, prompts["общий"])
    
    def create_training_data(self, max_examples: int = None) -> List[Dict[str, Any]]:
        training_data = []
        used_posts = set()
        
        system_message = {
            "role": "system",
            "content": "Ты специалист по маркетингу для бренда натуральных свечей ручной работы. Пишешь посты для Telegram канала в дружелюбном, экспертном тоне с использованием эмодзи. Специализируешься на натуральных свечах с парфюмированными ароматами, кокосовым воском и декором из сухоцветов и натуральных камней."
        }
        
        for post in self.posts:
            if len(post.strip()) < 50 or "навигация" in post.lower():
                continue
            
            post_content = post.strip()
            if post_content in used_posts:
                continue
            
            used_posts.add(post_content)
            
            category = self.categorize_post(post)
            prompts = self.generate_prompts(category)
            selected_prompt = prompts[0]
            
            training_example = {
                "messages": [
                    system_message,
                    {"role": "user", "content": selected_prompt},
                    {"role": "assistant", "content": post_content}
                ]
            }
            training_data.append(training_example)
            
            if max_examples and len(training_data) >= max_examples:
                print(f"🔄 Limited training data to {max_examples} examples to reduce cost")
                break
        
        print(f"Created {len(training_data)} unique training examples (removed duplicates)")
        return training_data
    
    def save_training_data(self, training_data: List[Dict[str, Any]], filename: str = "training_data.jsonl"):
        with open(filename, 'w', encoding='utf-8') as f:
            for example in training_data:
                f.write(json.dumps(example, ensure_ascii=False) + '\n')
        print(f"Saved {len(training_data)} training examples to {filename}")
        return filename

def upload_and_train(training_file: str, model_name: str = "gpt-3.5-turbo"):
    try:
        print("Uploading training file...")
        with open(training_file, 'rb') as f:
            upload_response = client.files.create(
                file=f,
                purpose='fine-tune'
            )
        
        file_id = upload_response.id
        print(f"File uploaded successfully. File ID: {file_id}")
        
        print("Starting fine-tuning job...")
        job_response = client.fine_tuning.jobs.create(
            training_file=file_id,
            model=model_name,
            hyperparameters={
                "n_epochs": 3,
            }
        )
        
        job_id = job_response.id
        print(f"Fine-tuning job started. Job ID: {job_id}")
        
        return job_id, file_id
        
    except Exception as e:
        print(f"Error during upload/training: {e}")
        return None, None

def monitor_training(job_id: str, timeout_minutes: int = 60):
    print("Monitoring training progress...")
    
    start_time = time.time()
    timeout_seconds = timeout_minutes * 60
    check_count = 0
    
    while True:
        try:
            check_count += 1
            elapsed_time = time.time() - start_time
            
            job_status = client.fine_tuning.jobs.retrieve(job_id)
            status = job_status.status
            
            print(f"Check #{check_count} | Elapsed: {elapsed_time/60:.1f}min | Status: {status}")
            
            if hasattr(job_status, 'error') and job_status.error:
                print(f"Error details: {job_status.error}")
            
            if hasattr(job_status, 'trained_tokens') and job_status.trained_tokens:
                print(f"Trained tokens: {job_status.trained_tokens}")
                
            if hasattr(job_status, 'training_file') and job_status.training_file:
                print(f"Training file: {job_status.training_file}")
            
            if status == "succeeded":
                model_id = job_status.fine_tuned_model
                print(f"Training completed successfully!")
                print(f"Fine-tuned model ID: {model_id}")
                return model_id
            elif status == "failed":
                print("Training failed!")
                if hasattr(job_status, 'error') and job_status.error:
                    print(f"Error: {job_status.error}")
                return None
            elif status in ["cancelled", "cancelled_by_user"]:
                print("Training was cancelled")
                return None
            elif status == "validating_files":
                if elapsed_time > 600:
                    print("File validation taking too long (>10min). This might indicate an issue.")
                    print("Consider checking your training data format or trying again.")
                    
                    print("Options:")
                    print("1. Continue waiting (training might still succeed)")
                    print("2. Cancel and check the data")
                    
                    if elapsed_time > 1800:
                        print("Validation timeout after 30 minutes. Likely an issue with the data.")
                        return None
            
            if elapsed_time > timeout_seconds:
                print(f"Training timeout after {timeout_minutes} minutes")
                print("You can check the job status later using the job ID:", job_id)
                return None
            
            if status == "validating_files":
                wait_time = 60
            elif status in ["queued", "running"]:
                wait_time = 30
            else:
                wait_time = 60
                
            print(f"Waiting {wait_time} seconds before next check...")
            time.sleep(wait_time)
            
        except Exception as e:
            print(f"Error checking status: {e}")
            print("Retrying in 60 seconds...")
            time.sleep(60)
            
            if 'error_count' not in locals():
                error_count = 1
            else:
                error_count += 1
                
            if error_count > 5:
                print("Too many errors checking job status. Giving up.")
                return None

def check_training_data_quality(training_data: List[Dict[str, Any]]) -> bool:
    print("🔍 Checking training data quality...")
    
    if len(training_data) < 10:
        print("⚠️ Warning: Very few training examples. Consider adding more data.")
        
    for i, example in enumerate(training_data[:5]):
        if "messages" not in example:
            print(f"Example {i} missing 'messages' field")
            return False
            
        messages = example["messages"]
        if len(messages) != 3:
            print(f"Example {i} should have exactly 3 messages (system, user, assistant)")
            return False
            
        roles = [msg["role"] for msg in messages]
        if roles != ["system", "user", "assistant"]:
            print(f"Example {i} has incorrect role sequence: {roles}")
            return False
        
        for j, msg in enumerate(messages):
            if len(msg["content"]) < 10:
                print(f"Example {i}, message {j} is very short: {len(msg['content'])} chars")
            elif len(msg["content"]) > 4000:
                print(f"Example {i}, message {j} is very long: {len(msg['content'])} chars")
    
    print(f"Training data format looks good! {len(training_data)} examples ready.")
    return True

def get_job_status(job_id: str):
    try:
        job_status = client.fine_tuning.jobs.retrieve(job_id)
        print(f"Job ID: {job_id}")
        print(f"Status: {job_status.status}")
        print(f"Model: {job_status.model}")
        print(f"Created: {job_status.created_at}")
        
        if hasattr(job_status, 'fine_tuned_model') and job_status.fine_tuned_model:
            print(f"Fine-tuned model: {job_status.fine_tuned_model}")
            
        if hasattr(job_status, 'error') and job_status.error:
            print(f"Error: {job_status.error}")
            
        return job_status
    except Exception as e:
        print(f"Error retrieving job status: {e}")
        return None

def test_model(model_id: str, test_prompts: List[str]):
    print(f"\nTesting fine-tuned model: {model_id}")
    
    system_message = "Ты специалист по маркетингу для бренда натуральных свечей ручной работы."
    
    for i, prompt in enumerate(test_prompts, 1):
        try:
            print(f"\n--- Test {i} ---")
            print(f"Prompt: {prompt}")
            
            response = client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.7
            )
            
            generated_text = response.choices[0].message.content
            print(f"Generated post:\n{generated_text}")
            print("-" * 50)
            
        except Exception as e:
            print(f"Error testing model: {e}")

def main():
    raw_posts_file = "/Users/yarik/Desktop/AromaKiss Project/messages_simple_list.json"
    
    if not os.path.exists(raw_posts_file):
        print(f"File {raw_posts_file} not found!")
        return
    
    print("Starting fine-tuning process for Russian candle posts...")
    
    processor = CandlePostProcessor(raw_posts_file)
    print(f"Loaded {len(processor.posts)} raw posts")
    
    training_data = processor.create_training_data()
    training_file = processor.save_training_data(training_data)
    job_id, file_id = upload_and_train(training_file)
    
    if job_id:
        model_id = monitor_training(job_id)
        
        if model_id:
            test_prompts = [
                "Напиши интересный факт о свечах",
                "Расскажи о новых ароматах в коллекции", 
                "Создай вдохновляющий пост о свечах",
                "Опиши преимущества кокосового воска"
            ]
            
            test_model(model_id, test_prompts)
            
            print(f"\nSuccess. Your fine-tuned model is ready: {model_id}")
            print("You can now use this model ID in your applications!")
        
    print("\nSummary:")
    print(f"- Processed {len(processor.posts)} original posts")
    print(f"- Created {len(training_data)} training examples")
    print(f"- Training file: {training_file}")
    if 'job_id' in locals():
        print(f"- Job ID: {job_id}")
    if 'model_id' in locals() and model_id:
        print(f"- Model ID: {model_id}")

if __name__ == "__main__":
    main()