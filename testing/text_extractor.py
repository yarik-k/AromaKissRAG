import json

def extract_text_only_messages(input_file_path, output_file_path='messages_text_only.json'):
    try:
        with open(input_file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        channel_info = data.get('channel_info', {})
        messages = data.get('messages', [])
        
        print(f"Processing channel: {channel_info.get('name', 'Unknown')}")
        print(f"Total messages found: {len(messages)}")
        
        text_messages = []
        messages_with_text = 0
        
        for message in messages:
            text_content = message.get('text', '').strip()
            
            if text_content:
                text_messages.append(text_content)
                messages_with_text += 1
        
        print(f"Messages with text content: {messages_with_text}")
        print(f"Empty messages skipped: {len(messages) - messages_with_text}")
        
        output_data = {
            "channel_info": {
                "name": channel_info.get('name', 'Unknown'),
                "id": channel_info.get('id', 'Unknown'),
                "total_messages": len(messages),
                "messages_with_text": messages_with_text
            },
            "messages_text": text_messages
        }
        
        with open(output_file_path, 'w', encoding='utf-8') as file:
            json.dump(output_data, file, ensure_ascii=False, indent=2)
        
        print(f"\nText-only messages saved to: {output_file_path}")
        
        print(f"\nPreview of first 3 messages:")
        print("-" * 50)
        for i, text in enumerate(text_messages[:3]):
            preview = text[:200] + "..." if len(text) > 200 else text
            print(f"{i+1}. {preview}")
            print()
        
        return text_messages
        
    except FileNotFoundError:
        print(f"Error: File '{input_file_path}' not found.")
        return []
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in '{input_file_path}'.")
        return []
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        return []

def extract_text_as_simple_list(input_file_path, output_file_path='messages_simple_list.json'):
    try:
        with open(input_file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        messages = data.get('messages', [])
        
        text_list = []
        
        for message in messages:
            text_content = message.get('text', '').strip()
            if text_content:
                text_list.append(text_content)
        
        with open(output_file_path, 'w', encoding='utf-8') as file:
            json.dump(text_list, file, ensure_ascii=False, indent=2)
        
        print(f"Simple text list saved to: {output_file_path}")
        print(f"Total text messages: {len(text_list)}")
        
        return text_list
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return []

def extract_text_as_single_string(input_file_path, output_file_path='messages_combined.txt'):
    try:
        with open(input_file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        messages = data.get('messages', [])
        channel_name = data.get('channel_info', {}).get('name', 'Unknown Channel')
        
        combined_text = f"Messages from: {channel_name}\n"
        combined_text += "=" * 50 + "\n\n"
        
        message_count = 0
        for message in messages:
            text_content = message.get('text', '').strip()
            if text_content:
                message_count += 1
                combined_text += f"Message {message_count}:\n"
                combined_text += text_content + "\n\n"
                combined_text += "-" * 30 + "\n\n"
        
        with open(output_file_path, 'w', encoding='utf-8') as file:
            file.write(combined_text)
        
        print(f"Combined text saved to: {output_file_path}")
        print(f"Total messages combined: {message_count}")
        
        return combined_text
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return ""

def main():
    print("Text-Only Message Extractor")
    print("=" * 40)
    
    input_file = 'extracted_messages.json'
    
    print("Choose output format:")
    print("1. JSON with metadata (default)")
    print("2. Simple JSON list")
    print("3. Combined text file")
    print("4. All formats")
    
    choice = input("\nEnter your choice (1-4, or press Enter for default): ").strip()
    
    if choice == '2':
        extract_text_as_simple_list(input_file)
    elif choice == '3':
        extract_text_as_single_string(input_file)
    elif choice == '4':
        print("\nGenerating all formats...")
        extract_text_only_messages(input_file)
        extract_text_as_simple_list(input_file)
        extract_text_as_single_string(input_file)
    else:
        extract_text_only_messages(input_file)

if __name__ == "__main__":
    main()