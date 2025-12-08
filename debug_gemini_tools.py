
import sys
import google.generativeai as genai

with open('debug_output.txt', 'w') as f:
    sys.stdout = f
    try:
        if hasattr(genai.protos.Tool, 'DESCRIPTOR'):
            field = genai.protos.Tool.DESCRIPTOR.fields_by_name.get('google_search')
            if field:
                print("google_search field type:", field.message_type.name)
            else:
                print("google_search field NOT found")
            
            field = genai.protos.Tool.DESCRIPTOR.fields_by_name.get('google_search_retrieval')
            if field:
                print("google_search_retrieval field type:", field.message_type.name)
            else:
                print("google_search_retrieval field NOT found")
                
    except Exception as e:
        print("Error:", e)
