# KG-RAG POC for Obesity Management

A Knowledge Graph-based Retrieval-Augmented Generation (KG-RAG) system for the UNIFIED project, demonstrating integration of Digital Health Technologies (DHT), Clinical Outcome Assessments (COA), and Patient Preference Information (PPI).

## Features
- Knowledge Graph modeling using NetworkX
- RAG-based context retrieval with explainable reasoning chains
- Simulated LLM response generation
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

## Usage
```powershell
python main.py
```

## Output
The system provides:
1. **LLM Conclusion**: Clinical explanation of the patient's condition
2. **Explainability Evidence**: Knowledge Graph paths supporting the conclusion

## Files
- `kg_model.py`: Knowledge Graph schema and construction
- `rag_engine.py`: Retrieval and reasoning logic
- `main.py`: Main simulation script
