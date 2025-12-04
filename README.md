# KG-RAG POC for Obesity Management

A Knowledge Graph-based Retrieval-Augmented Generation (KG-RAG) system for the UNIFIED project, demonstrating integration of Digital Health Technologies (DHT), Clinical Outcome Assessments (COA), and Patient Preference Information (PPI).

## Features
- **Knowledge Graph modeling** using NetworkX
- **RAG-based context retrieval** with explainable reasoning chains
- **LangChain integration** with optional LLM API support (OpenAI, Google Gemini)
- **Multiple patient scenarios** (3 distinct obesity management cases)
- **Graph visualization** with matplotlib (full graph + reasoning chains)
- **Comprehensive unit tests** with pytest
- Focus on obesity management use case

## Setup

### 1. Create Virtual Environment
```powershell
python -m venv venv
```

### 2. Activate Virtual Environment
```powershell
.\venv\Scripts\Activate.ps1
```

### 3. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 4. (Optional) Configure LLM API Keys
Copy `.env.example` to `.env` and add your API keys:
```powershell
cp .env.example .env
# Edit .env with your favorite editor
```

If no API keys are provided, the system will use simulated LLM responses.

## Usage

### Run Full Simulation
Processes all 3 patient scenarios and generates visualizations:
```powershell
python main.py
```

### Run Unit Tests
```powershell
python -m pytest test_kg_rag.py -v
```

### Generate Visualizations Only
```powershell
python visualize.py
```

## Output
The system provides for each patient:
1. **LLM Conclusion**: Clinical explanation with actionable recommendations
2. **Explainability Evidence**: Knowledge Graph paths (nodes and edges) supporting the conclusion
3. **Visual Reasoning Chain**: PNG image highlighting the conflict path

Generated files:
- `full_knowledge_graph.png` - Complete KG visualization
- `reasoning_chain_Patient_A.png` - Patient A's reasoning path
- `reasoning_chain_Patient_B.png` - Patient B's reasoning path  
- `reasoning_chain_Patient_C.png` - Patient C's reasoning path

## Patient Scenarios

### Patient A: Diet Flexibility Conflict
- **Issue**: Slow weight loss despite moderate activity
- **Root Cause**: Preference for diet flexibility conflicts with calorie management
- **KG Path**: `Diet_Flexibility_Preference --[conflicts_with]--> Calorie_Intake --[influences]--> Slow_Weight_Loss`

### Patient B: Exercise Compensation
- **Issue**: High exercise but no weight loss
- **Root Cause**: Minimal diet awareness leads to compensatory eating
- **KG Path**: `Minimal_Diet_Preference --[conflicts_with]--> Calorie_Intake --[influences]--> Slow_Weight_Loss`

### Patient C: Metabolic Adaptation
- **Issue**: Weight plateau despite excellent adherence
- **Root Cause**: Strict routine causes metabolic adaptation
- **KG Path**: `Strict_Adherence_Preference --[conflicts_with]--> Metabolic_Adaptation --[causes]--> Weight_Plateau`

## Project Structure
```
kg_poc/
├── kg_model.py           # Knowledge Graph schema and construction
├── rag_engine.py         # Retrieval and reasoning logic
├── visualize.py          # Graph visualization functions
├── main.py               # Main orchestration with LangChain
├── test_kg_rag.py        # Unit tests
├── requirements.txt      # Python dependencies
├── .env.example          # Environment variables template
└── README.md             # This file
```

## Technologies
- **NetworkX**: Graph data structure
- **LangChain**: LLM orchestration framework
- **Matplotlib**: Visualization
- **Pytest**: Testing framework
- **Python-dotenv**: Environment management
