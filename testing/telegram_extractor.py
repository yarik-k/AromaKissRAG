import json
import csv
from datetime import datetime

def extract_text_from_entities(text_data):
    if isinstance(text_data, str):
        return text_data
    
    if isinstance(text_data, list):
        extracted_text = ""
        for entity in text_data:
            if isinstance(entity, dict):
                if 'text' in entity:
                    extracted_text += entity['text']
            elif isinstance(entity, str):
                extracted_text += entity
        return extracted_text
    
    return str(text_data) if text_data else ""

def extract_messages_from_telegram_export(json_file_path, output_format='json'):
    try:
        with open(json_file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        channel_name = data.get('name', 'Unknown Channel')
        channel_id = data.get('id', 'Unknown ID')
        
        print(f"Processing channel: {channel_name}")
        print(f"Channel ID: {channel_id}")
        
        messages = data.get('messages', [])
        extracted_messages = []
        
        for message in messages:
            if message.get('type') == 'service':
                continue
            
            msg_data = {
                'id': message.get('id'),
                'date': message.get('date'),
                'date_unixtime': message.get('date_unixtime'),
                'type': message.get('type'),
                'from': message.get('from'),
                'from_id': message.get('from_id'),
                'text': extract_text_from_entities(message.get('text', '')),
                'edited': message.get('edited'),
                'edited_unixtime': message.get('edited_unixtime'),
                'has_photo': 'photo' in message,
                'photo_info': message.get('photo', '') if 'photo' in message else None,
                'reactions': message.get('reactions', [])
            }
            
            if msg_data['date_unixtime']:
                try:
                    readable_date = datetime.fromtimestamp(int(msg_data['date_unixtime']))
                    msg_data['readable_date'] = readable_date.strftime('%Y-%m-%d %H:%M:%S')
                except:
                    msg_data['readable_date'] = msg_data['date']
            else:
                msg_data['readable_date'] = msg_data['date']
            
            extracted_messages.append(msg_data)
        
        print(f"Found {len(extracted_messages)} regular messages (excluding service messages)")
        
        if output_format.lower() == 'json':
            output_file = 'extracted_messages.json'
            with open(output_file, 'w', encoding='utf-8') as file:
                json.dump({
                    'channel_info': {
                        'name': channel_name,
                        'id': channel_id,
                        'total_messages': len(extracted_messages)
                    },
                    'messages': extracted_messages
                }, file, ensure_ascii=False, indent=2)
            print(f"Messages saved to {output_file}")
        
        elif output_format.lower() == 'csv':
            output_file = 'extracted_messages.csv'
            with open(output_file, 'w', newline='', encoding='utf-8') as file:
                if extracted_messages:
                    fieldnames = ['id', 'readable_date', 'date', 'type', 'from', 'text', 'has_photo', 'edited']
                    writer = csv.DictWriter(file, fieldnames=fieldnames)
                    writer.writeheader()
                    
                    for msg in extracted_messages:
                        row = {
                            'id': msg['id'],
                            'readable_date': msg['readable_date'],
                            'date': msg['date'],
                            'type': msg['type'],
                            'from': msg['from'],
                            'text': msg['text'][:500] + '...' if len(msg['text']) > 500 else msg['text'],  # Truncate long text
                            'has_photo': msg['has_photo'],
                            'edited': msg['edited'] or ''
                        }
                        writer.writerow(row)
            print(f"Messages saved to {output_file}")
        
        elif output_format.lower() == 'txt':
            output_file = 'extracted_messages.txt'
            with open(output_file, 'w', encoding='utf-8') as file:
                file.write(f"Channel: {channel_name}\n")
                file.write(f"Channel ID: {channel_id}\n")
                file.write(f"Total Messages: {len(extracted_messages)}\n")
                file.write("=" * 50 + "\n\n")
                
                for msg in extracted_messages:
                    file.write(f"Message ID: {msg['id']}\n")
                    file.write(f"Date: {msg['readable_date']}\n")
                    file.write(f"From: {msg['from']}\n")
                    if msg['has_photo']:
                        file.write("Has Photo\n")
                    if msg['edited']:
                        file.write(f"Edited: {msg['edited']}\n")
                    file.write(f"Text: {msg['text']}\n")
                    if msg['reactions']:
                        reactions_str = ", ".join([f"{r['emoji']}({r['count']})" for r in msg['reactions']])
                        file.write(f"Reactions: {reactions_str}\n")
                    file.write("-" * 30 + "\n\n")
            print(f"Messages saved to {output_file}")
        
        return extracted_messages
    
    except FileNotFoundError:
        print(f"Error: File '{json_file_path}' not found.")
        return []
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in '{json_file_path}'.")
        return []
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        return []

def main():
    json_file_path = '/Users/yarik/Desktop/AromaKiss Project/data/result.json'
    output_format = 'json'
    
    print("Telegram Message Extractor")
    print("=" * 30)
    
    messages = extract_messages_from_telegram_export(json_file_path, output_format)
    
    if messages:
        print(f"\nExtraction completed successfully!")
        print(f"Total messages extracted: {len(messages)}")
        
        print(f"\nPreview of first 3 messages:")
        print("-" * 40)
        for i, msg in enumerate(messages[:3]):
            print(f"{i+1}. ID: {msg['id']}")
            print(f"   Date: {msg['readable_date']}")
            print(f"   Text: {msg['text'][:100]}{'...' if len(msg['text']) > 100 else ''}")
            print()
    else:
        print("No messages were extracted.")

if __name__ == "__main__":
    main()