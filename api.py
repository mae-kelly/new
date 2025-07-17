from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, List, Any
import json

app = FastAPI(title="Simple Visibility ML API")

try:
    with open('models/results.json', 'r') as f:
        sample_results = json.load(f)
except:
    sample_results = {'mappings': [], 'total_mappings': 0}

class DataRequest(BaseModel):
    tables: Dict[str, List[Dict[str, Any]]]

@app.get("/")
def root():
    return {"status": "Simple ML API Ready", "sample_mappings": len(sample_results['mappings'])}

@app.post("/analyze")
def analyze(request: DataRequest):
    from simple_ml import ml_engine
    
    all_mappings = []
    for table_name, table_data in request.tables.items():
        mappings = ml_engine.analyze_table(table_name, table_data)
        all_mappings.extend(mappings)
    
    all_mappings.sort(key=lambda x: x['confidence'], reverse=True)
    
    return {
        "mappings": all_mappings,
        "total_mappings": len(all_mappings),
        "high_confidence": len([m for m in all_mappings if m['confidence'] > 0.7]),
        "confidence_distribution": {
            "high": len([m for m in all_mappings if m['confidence'] > 0.8]),
            "medium": len([m for m in all_mappings if 0.5 <= m['confidence'] <= 0.8]),
            "low": len([m for m in all_mappings if m['confidence'] < 0.5])
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
