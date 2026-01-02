#!/usr/bin/env python3

import json
import os
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
import openai
from sklearn.metrics.pairwise import cosine_similarity
import re
import logging
from datetime import datetime

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class RetrievedPost:
    content: str
    similarity: float
    post_type: str
    keywords: List[str]

class AromaKissRAG:
    def __init__(self, openai_api_key: str, messages_file: str = "messages_simple_list.json"):
        self.openai_client = openai.OpenAI(api_key=openai_api_key)
        self.messages_file = messages_file
        
        logger.info("Loading multilingual sentence transformer...")
        self.encoder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        
        self.posts = []
        self.embeddings = None
        self.post_metadata = []
        
        self._load_and_process_messages()
        self._create_embeddings()
        
        logger.info(f"RAG system initialized with {len(self.posts)} posts")
    
    def _load_and_process_messages(self):
        try:
            with open(self.messages_file, 'r', encoding='utf-8') as f:
                self.posts = json.load(f)
            
            for i, post in enumerate(self.posts):
                metadata = self._analyze_post(post)
                self.post_metadata.append(metadata)
                
        except FileNotFoundError:
            logger.error(f"Messages file {self.messages_file} not found")
            raise
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in {self.messages_file}")
            raise
    
    def _analyze_post(self, post: str) -> Dict:
        metadata = {
            'length': len(post),
            'has_emoji': bool(re.search(r'[😀-🙏🌀-🗿🚀-🛿]', post)),
            'has_hashtag': '#' in post,
            'post_type': 'general',
            'topics': [],
            'season': None,
            'sentiment': 'neutral'
        }
        
        post_lower = post.lower()
        
        if any(word in post_lower for word in ['интересн', 'факт']):
            metadata['post_type'] = 'educational'
        elif any(word in post_lower for word in ['новогод', 'рождеств', '8 марта', 'весн']):
            metadata['post_type'] = 'seasonal'
        elif any(word in post_lower for word in ['аромат', 'запах', 'парфюм']):
            metadata['post_type'] = 'fragrance'
        elif any(word in post_lower for word in ['декор', 'сухоцвет', 'камн']):
            metadata['post_type'] = 'decor'
        elif any(word in post_lower for word in ['заказ', 'подарок', 'цена']):
            metadata['post_type'] = 'commercial'
        elif any(word in post_lower for word in ['процесс', 'создан', 'изготовл']):
            metadata['post_type'] = 'process'
        
        topics = []
        topic_keywords = {
            'ароматы': ['аромат', 'запах', 'парфюм', 'отдушк'],
            'декор': ['декор', 'сухоцвет', 'камн', 'украшен'],
            'процесс': ['процесс', 'создан', 'изготовл', 'ручн'],
            'материалы': ['воск', 'кокосов', 'натуральн', 'качеств'],
            'праздники': ['новогод', 'рождеств', '8 марта', 'праздник'],
            'подарки': ['подарок', 'подар', 'заказ', 'сюрприз']
        }
        
        for topic, keywords in topic_keywords.items():
            if any(keyword in post_lower for keyword in keywords):
                topics.append(topic)
        
        metadata['topics'] = topics
        
        if any(word in post_lower for word in ['новогод', 'рождеств', 'зим']):
            metadata['season'] = 'winter'
        elif any(word in post_lower for word in ['весн', '8 марта']):
            metadata['season'] = 'spring'
        elif any(word in post_lower for word in ['лет']):
            metadata['season'] = 'summer'
        elif any(word in post_lower for word in ['осен']):
            metadata['season'] = 'autumn'
        
        return metadata
    
    def _create_embeddings(self):
        logger.info("Creating embeddings for posts...")
        self.embeddings = self.encoder.encode(self.posts, show_progress_bar=True)
        logger.info("Embeddings created successfully")
    
    def _retrieve_similar_posts(self, query: str, num_posts: int = 5, 
                              post_type_filter: Optional[str] = None) -> List[RetrievedPost]:
        query_embedding = self.encoder.encode([query])
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        sorted_indices = np.argsort(similarities)[::-1]
        
        retrieved_posts = []
        for idx in sorted_indices:
            if len(retrieved_posts) >= num_posts:
                break
                
            if post_type_filter and self.post_metadata[idx]['post_type'] != post_type_filter:
                continue
            
            retrieved_post = RetrievedPost(
                content=self.posts[idx],
                similarity=similarities[idx],
                post_type=self.post_metadata[idx]['post_type'],
                keywords=self.post_metadata[idx]['topics']
            )
            retrieved_posts.append(retrieved_post)
        
        return retrieved_posts
    
    def _create_system_prompt(self, task_type: str) -> str:
        base_persona = """Ты - основательница премиального бренда свечей ручной работы. Ты создаёшь роскошные свечи с ароматами культовых парфюмов.

ТВОЯ ЛИЧНОСТЬ:
- Элегантная, тёплая и эмоционально вовлекающая
- Страстно увлечена своим делом
- Используешь эмодзи стратегически (💋, 🕯, ✨, 🥰, 🌺)
- Пишешь с душой и для души

БРЕНД:
- Роскошные свечи ручной работы на кокосовом воске
- Эксклюзивные парфюмерные отдушки из Европы
- Натуральный декор (сухоцветы, драгоценные камни)
- Индивидуальный подход к каждому заказу
- Время изготовления: 4-6 дней
- Также создаёшь изысканные аромадиффузоры

СТИЛЬ ПИСЬМА:
- Начинаешь с эмодзи или цепляющего крючка
- Используешь короткие абзацы с переносами строк
- Включаешь релевантные хештеги
- Заканчиваешь тепло, часто фирменными фразами
- Сочетаешь информацию о продукте с lifestyle-контентом"""

        if task_type == "post_writing":
            return base_persona + """

ЗАДАЧА: Напиши пост для Telegram-канала, используя примеры как референс для тона, структуры и манеры изложения. Сохраняй аутентичность и страсть к созданию прекрасных ароматических впечатлений."""

        elif task_type == "idea_generation":
            return base_persona + """

ЗАДАЧА: Генерируй креативные идеи для постов, основываясь на успешных паттернах из примеров. Предлагай разнообразные темы: образовательные, сезонные, продуктовые, эмоциональные, интерактивные."""

        elif task_type == "research":
            return base_persona + """

ЗАДАЧА: Проводи исследования для создания контента о свечах, ароматах, традициях и всём, что связано с миром свечей. Используй примеры как основу для понимания интересов аудитории и стиля подачи информации."""

        elif task_type == "conversation":
            return base_persona + """

ЗАДАЧА: Веди естественную беседу. Анализируй контекст разговора и реагируй соответственно:

1. **Если пользователь просит изменить/улучшить предыдущий контент** - внимательно изучи историю разговора, найди что нужно изменить, и внеси запрашиваемые правки, сохраняя свой стиль.

2. **Если пользователь задает новый вопрос или меняет тему** - отвечай дружелюбно и тепло. Можешь делиться личными мыслями, опытом, советами.

3. **Если разговор касается свечей, ароматов или творчества** - с удовольствием рассказывай подробнее, но не превращай каждый ответ в рекламу.

Будь внимательной к контексту и естественной в общении. Если неясно, что именно пользователь хочет изменить в предыдущем ответе, вежливо уточни."""

        elif task_type == "refinement":
            return base_persona + """

ЗАДАЧА: Ты получаешь запрос на изменение или улучшение ранее созданного контента. Внимательно изучи историю разговора, найди что именно нужно изменить, и внеси запрашиваемые правки. Сохраняй свой стиль и качество. Возможные типы изменений:
- Сделать короче/длиннее
- Изменить тон (формальнее/неформальнее)
- Добавить/убрать детали
- Переписать в другом стиле
- Исправить или улучшить содержание
Если запрос неясен, вежливо уточни что именно нужно изменить."""

        return base_persona

    def generate_post(self, user_request: str, style_examples: int = 4, conversation_context: str = "") -> str:
        logger.info(f"Generating post for request: {user_request[:50]}...")
        
        similar_posts = self._retrieve_similar_posts(user_request, style_examples)
        
        examples_text = "\n\n--- ПРИМЕРЫ ТВОИХ ПОСТОВ ---\n"
        for i, post in enumerate(similar_posts, 1):
            examples_text += f"\nПример {i} (схожесть: {post.similarity:.2f}):\n{post.content}\n"
        
        system_prompt = self._create_system_prompt("post_writing")
        context_text = conversation_context if conversation_context else ""
        user_prompt = f"{context_text}{examples_text}\n\n--- ЗАДАНИЕ ---\nНапиши пост на тему: {user_request}\n\nИспользуй примеры выше как референс твоего стиля, тона и манеры изложения. Пиши естественно и аутентично."
        
        response = self.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.8,
            max_tokens=1000
        )
        
        return response.choices[0].message.content
    
    def generate_post_ideas(self, theme: str = "", num_ideas: int = 5, conversation_context: str = "") -> str:
        logger.info(f"Generating {num_ideas} post ideas for theme: {theme}")
        
        if theme:
            similar_posts = self._retrieve_similar_posts(theme, 6)
        else:
            similar_posts = []
            post_types = ['educational', 'seasonal', 'fragrance', 'decor', 'commercial']
            for post_type in post_types:
                posts = self._retrieve_similar_posts("свечи", 2, post_type)
                similar_posts.extend(posts)
        
        examples_text = "\n\n--- УСПЕШНЫЕ ПОСТЫ ДЛЯ ВДОХНОВЕНИЯ ---\n"
        for i, post in enumerate(similar_posts, 1):
            examples_text += f"\nПост {i} ({post.post_type}):\n{post.content}\n"
        
        system_prompt = self._create_system_prompt("idea_generation")
        context_text = conversation_context if conversation_context else ""
        theme_text = f" на тему '{theme}'" if theme else ""
        user_prompt = f"{context_text}{examples_text}\n\n--- ЗАДАНИЕ ---\nПредложи {num_ideas} креативных идей для постов{theme_text}.\n\nОсновывайся на успешных паттернах из примеров выше. Каждая идея должна включать:\n- Заголовок/тему\n- Краткое описание содержания\n- Предполагаемый стиль подачи\n- Возможные эмодзи и хештеги"
        
        response = self.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.9,
            max_tokens=1200
        )
        
        return response.choices[0].message.content
    
    def research_topic(self, research_query: str, conversation_context: str = "") -> str:
        logger.info(f"Researching topic: {research_query}")
        
        relevant_posts = self._retrieve_similar_posts(research_query, 4)
        
        context_text = "\n\n--- КОНТЕКСТ ИЗ ТВОИХ ПОСТОВ ---\n"
        for i, post in enumerate(relevant_posts, 1):
            context_text += f"\nПост {i}:\n{post.content}\n"
        
        system_prompt = self._create_system_prompt("research")
        conversation_text = conversation_context if conversation_context else ""
        user_prompt = f"{conversation_text}{context_text}\n\n--- ИССЛЕДОВАНИЕ ---\nИсследуй тему: {research_query}\n\nПредоставь полезную информацию, которую можно использовать для создания интересного и образовательного поста. Включи:\n- Интересные факты\n- Историческую информацию\n- Практические советы\n- Связь с ароматерапией/свечами\n- Идеи для креативной подачи\n\nОснуйся на контексте из моих существующих постов и дополни новой полезной информацией."
        
        response = self.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            max_tokens=1500
        )
        
        return response.choices[0].message.content
    
    def refine_content(self, refinement_request: str, conversation_context: str, content_type: str = "general") -> str:
        logger.info(f"Refining {content_type} content: {refinement_request[:50]}...")
        
        system_prompt = self._create_system_prompt("refinement")
        user_prompt = f"{conversation_context}\n\n--- ЗАПРОС НА ИЗМЕНЕНИЕ ---\nПользователь просит: {refinement_request}\n\nПроанализируй предыдущий разговор и найди контент, который нужно изменить. Внеси запрашиваемые изменения, сохраняя мой стиль и качество. Если нужно изменить пост, идеи или исследование - сделай это. Если просьба неясна, уточни что именно нужно изменить."
        
        response = self.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.8,
            max_tokens=1200
        )
        
        return response.choices[0].message.content
    
    def conversational_chat(self, user_message: str, conversation_context: str = "") -> str:
        logger.info(f"Processing conversational message: {user_message[:50]}...")
        
        system_prompt = self._create_system_prompt("conversation")
        user_prompt = f"{conversation_context}\n\n--- ТЕКУЩЕЕ СООБЩЕНИЕ ---\nПользователь: {user_message}\n\nПроанализируй контекст разговора. Если пользователь просит изменить или улучшить предыдущий ответ - сделай это. Если это новый вопрос или тема - ответь естественно и дружелюбно."
        
        response = self.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.9,
            max_tokens=800
        )
        
        return response.choices[0].message.content
    
    def interactive_session(self):
        print("🕯 Добро пожаловать в RAG Bot! 🕯")
        print("Доступные команды:")
        print("1. 'пост: [описание]' - генерация поста")
        print("2. 'идеи: [тема]' - генерация идей для постов")
        print("3. 'исследование: [тема]' - исследование темы")
        print("4. 'выход' - завершить сессию")
        print("-" * 50)
        
        while True:
            try:
                user_input = input("\n💫 Ваш запрос: ").strip()
                
                if user_input.lower() in ['выход', 'exit', 'quit']:
                    print("До свидания! 💋")
                    break
                
                if user_input.startswith('пост:'):
                    request = user_input[5:].strip()
                    if request:
                        result = self.generate_post(request)
                        print(f"\n📝 Сгенерированный пост:\n{result}")
                    else:
                        print("Пожалуйста, укажите тему поста после 'пост:'")
                
                elif user_input.startswith('идеи:'):
                    theme = user_input[5:].strip()
                    result = self.generate_post_ideas(theme)
                    print(f"\n💡 Идеи для постов:\n{result}")
                
                elif user_input.startswith('исследование:'):
                    topic = user_input[12:].strip()
                    if topic:
                        result = self.research_topic(topic)
                        print(f"\n🔍 Результаты исследования:\n{result}")
                    else:
                        print("Пожалуйста, укажите тему исследования после 'исследование:'")
                
                else:
                    print("Неизвестная команда. Используйте 'пост:', 'идеи:', 'исследование:' или 'выход'")
                    
            except KeyboardInterrupt:
                print("\n\nДо свидания! 💋")
                break
            except Exception as e:
                logger.error(f"Error in interactive session: {e}")
                print(f"Произошла ошибка: {e}")

def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY environment variable is not set.")
        api_key = input("Введите ваш OpenAI API ключ: ").strip()
        if not api_key:
            print("API ключ обязателен для работы бота")
            print("Установите переменную окружения OPENAI_API_KEY или создайте файл .env")
            return
    
    try:
        rag_bot = AromaKissRAG(api_key)
        rag_bot.interactive_session()
        
    except Exception as e:
        logger.error(f"Failed to initialize RAG bot: {e}")
        print(f"Ошибка инициализации: {e}")

if __name__ == "__main__":
    main() 