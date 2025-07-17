import json
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.parse

try:
    with open('models/results.json', 'r') as f:
        sample_results = json.load(f)
except:
    sample_results = {'mappings': [], 'total_mappings': 0}

class SimpleAPIHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        
        response = {
            "status": "Simple ML API Ready", 
            "sample_mappings": len(sample_results['mappings']),
            "endpoints": ["/", "/analyze"]
        }
        self.wfile.write(json.dumps(response, indent=2).encode())
    
    def do_POST(self):
        if self.path == '/analyze':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            
            try:
                request_data = json.loads(post_data.decode('utf-8'))
                tables = request_data.get('tables', {})
                
                from simple_ml import ml_engine
                
                all_mappings = []
                for table_name, table_data in tables.items():
                    mappings = ml_engine.analyze_table(table_name, table_data)
                    all_mappings.extend(mappings)
                
                all_mappings.sort(key=lambda x: x['confidence'], reverse=True)
                
                response = {
                    "mappings": all_mappings,
                    "total_mappings": len(all_mappings),
                    "high_confidence": len([m for m in all_mappings if m['confidence'] > 0.7]),
                    "confidence_distribution": {
                        "high": len([m for m in all_mappings if m['confidence'] > 0.8]),
                        "medium": len([m for m in all_mappings if 0.5 <= m['confidence'] <= 0.8]),
                        "low": len([m for m in all_mappings if m['confidence'] < 0.5])
                    }
                }
                
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(response, indent=2).encode())
                
            except Exception as e:
                self.send_response(400)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                error_response = {"error": str(e)}
                self.wfile.write(json.dumps(error_response).encode())
        else:
            self.send_response(404)
            self.end_headers()

if __name__ == "__main__":
    server = HTTPServer(('localhost', 8000), SimpleAPIHandler)
    print("🚀 Simple API server running on http://localhost:8000")
    print("Endpoints:")
    print("  GET  / - Status")
    print("  POST /analyze - Analyze data")
    server.serve_forever()
