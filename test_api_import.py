try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    print("SUCCESS: langchain_google_genai imported")
except ImportError as e:
    print(f"FAILURE: langchain_google_genai not found: {e}")

try:
    from langchain_openai import ChatOpenAI
    print("SUCCESS: langchain_openai imported")
except ImportError as e:
    print(f"FAILURE: langchain_openai not found: {e}")
