from label_studio_sdk import Client
import os
import time

# Get from Account & Settings → Access Token
PROJECT_ID = 1  # Your project ID (check URL or project settings)

# Try different API key formats
api_key_options = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ0b2tlbl90eXBlIjoicmVmcmVzaCIsImV4cCI6ODA2NjkwMDQ0MywiaWF0IjoxNzU5NzAwNDQzLCJqdGkiOiIwMGM2YzliNWRmMWM0YjI0YjAwZmNmNjZmMGIwMzE3OSIsInVzZXJfaWQiOjF9.mPZIqoWro6dKVKT8PewnZ7rIvgV1NkXgVTC2IeypGt4'

ls_host = "http://localhost:9001"

# Try different API key formats
ls = None
project = None
api_key_used = None

for i, api_key in enumerate(api_key_options):
    try:
        print(f"Trying API key format {i+1}...")
        ls = Client(url=ls_host, api_key=api_key)
        project = ls.get_project(PROJECT_ID)
        api_key_used = api_key
        print(f"✓ Connected to Label Studio project: {project.title}")
        break
    except Exception as e:
        print(f"✗ API key format {i+1} failed: {e}")
        continue

if project is None:
    print("\n❌ All API key formats failed!")
    print("\nPlease try one of these solutions:")
    print("1. Get a new API key from Label Studio:")
    print("   - Go to Account & Settings → Access Token")
    print("   - Generate a new token")
    print("2. Check if Label Studio is running:")
    print("   - Visit http://localhost:9001 in your browser")
    print("3. Try using username/password instead:")
    print("   - Replace api_key with your Label Studio username/password")
    exit(1)

try:
    
    # Get tasks with pagination to handle large datasets
    tasks = project.get_tasks()
    print(f"Total tasks: {len(tasks)}")
    
    converted = 0
    skipped = 0
    errors = 0
    
    for i, task in enumerate(tasks):
        try:
            # Check if task already has annotations
            if task.get('annotations') and len(task.get('annotations', [])) > 0:
                skipped += 1
                continue
                
            # Check if task has predictions
            if not task.get('predictions') or len(task.get('predictions', [])) == 0:
                skipped += 1
                continue
                
            prediction = task['predictions'][0]
            
            # Create annotation from prediction
            annotation_data = {
                'result': prediction['result'],
                'lead_time': prediction.get('lead_time', 0),
                'created_at': prediction.get('created_at'),
                'updated_at': prediction.get('updated_at')
            }
            
            # Create the annotation
            project.create_annotation(
                task_id=task['id'],
                result=prediction['result']
            )
            converted += 1
            
            # Progress reporting
            if converted % 10 == 0:
                print(f"Processed {i+1}/{len(tasks)} tasks, converted {converted}, skipped {skipped}")
                
            # Small delay to avoid overwhelming the server
            time.sleep(0.1)
            
        except Exception as e:
            errors += 1
            print(f"Error processing task {task.get('id', 'unknown')}: {e}")
            continue
    
    print(f"\nConversion completed!")
    print(f"✓ Converted: {converted} predictions to annotations")
    print(f"⚠️  Skipped: {skipped} tasks (already annotated or no predictions)")
    print(f"❌ Errors: {errors} tasks failed")
    
except Exception as e:
    print(f"Failed to connect to Label Studio: {e}")
    print("Please check:")
    print("1. Label Studio is running on http://localhost:9001")
    print("2. API key is correct")
    print("3. Project ID is correct")