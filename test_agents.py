
import os
import sys

# Mocking streamlit to avoid errors in imported modules
class MockSt:
    def info(self, text): print(f"[ST INFO] {text}")
    def success(self, text): print(f"[ST SUCCESS] {text}")
    def warning(self, text): print(f"[ST WARNING] {text}")
    def error(self, text): print(f"[ST ERROR] {text}")
    def write(self, text): print(f"[ST WRITE] {text}")
    def json(self, data): print(f"[ST JSON] {data}")
    def divider(self): print("---")
    def subheader(self, text): print(f"## {text}")
    def header(self, text): print(f"# {text}")
    def markdown(self, text, **kwargs): print(f"{text}")
    class status:
        def __init__(self, label, expanded=True): pass
        def __enter__(self): return self
        def __exit__(self, exc_type, exc_val, exc_tb): pass
        def update(self, **kwargs): pass
    def expander(self, label):
        class MockExpander:
            def __enter__(self): return self
            def __exit__(self, *args): pass
        return MockExpander()
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def toggle(self, label, value=False, help=None): return value
    def columns(self, spec): return [MockSt() for _ in range(spec)] if isinstance(spec, int) else [MockSt() for _ in spec]

# Inject mock into sys.modules
sys.modules['streamlit'] = MockSt()

from video_agents.manager import ManagerAgent

def test_pipeline():
    video_path = r"c:\Users\Aniket Roy\Downloads\PPT_Video - Made with Clipchamp.mp4"
    if not os.path.exists(video_path):
        print(f"File not found: {video_path}")
        return

    print(f"Starting test for: {video_path}")
    
    # Simulate Turbo Mode config
    config = {
        "audio_model": "tiny",
        "llm_model": "microsoft/phi-1_5"
    }
    
    manager = ManagerAgent(config)
    
    # Wrap file in an object that has a .read() method
    class MockFile:
        def __init__(self, path):
            self.path = path
        def read(self):
            with open(self.path, "rb") as f:
                return f.read()
    
    try:
        report = manager.process_video(MockFile(video_path))
        print("\n" + "="*50)
        print("FINAL REPORT GENERATED SUCCESSFULLY")
        print("="*50)
        print(f"TITLE: {report.get('title')}")
        print(f"NARRATIVE: {report.get('narrative')[:500]}...")
        print("="*50)
    except Exception as e:
        print(f"\nCRITICAL ERROR DURING PIPELINE: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_pipeline()
