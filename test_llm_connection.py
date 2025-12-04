import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

# Load environment variables
load_dotenv()

def test_google_api():
    api_key = os.getenv("GOOGLE_API_KEY")
    print(f"Checking GOOGLE_API_KEY: {'Found' if api_key else 'Not Found'}")
    
    if not api_key:
        print("Please set GOOGLE_API_KEY in .env file")
        return

    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        print("Imported ChatGoogleGenerativeAI successfully")
        
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7, google_api_key=api_key)
        print("Initialized LLM with model='gemini-2.0-flash'")
        
        print("Sending test request...")
        response = llm.invoke([HumanMessage(content="Hello, are you working?")])
        print(f"Response received: {response.content}")
        print("SUCCESS: Google Gemini API is working!")
        
    except Exception as e:
        print(f"FAILURE: Google API error: {e}")

if __name__ == "__main__":
    test_google_api()
