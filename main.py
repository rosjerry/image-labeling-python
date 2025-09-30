import os
from dotenv import load_dotenv
from label_studio_sdk.client import LabelStudio

load_dotenv()

api_key = os.getenv("API_KEY")
ls_host = os.getenv("LABEL_STUDIO_URL")
ls = LabelStudio(base_url=ls_host, api_key=api_key)


# change these varibles with your needs
make = "honda"
model = "accord"
year = "2014"
color = "blue"
fuel_type = "petroleum(gas_in_us)"
steer_wheel = "left"


def create_prediction_data(known_labels):
    """
    Create prediction data structure for Label Studio
    """
    prediction_result = []

    label_mappings = {
        "make": known_labels.get("make"),
        "model": known_labels.get("model"),
        "color": known_labels.get("color"),
        "year": known_labels.get("year"),
        "steer_wheel": known_labels.get("steer_wheel"),
        "fuel_type": known_labels.get("fuel_type"),
    }

    for field_name, value in label_mappings.items():
        if value:
            prediction_result.append(
                {
                    "value": {"choices": [value]},
                    "from_name": field_name,
                    "to_name": "image",
                    "type": "choices",
                }
            )

    return prediction_result


def bulk_update_tasks(project_id, known_labels):
    """
    Update all tasks in a project with known labels as predictions
    """
    try:
        print(f"Fetching tasks from project {project_id}...")
        tasks = ls.tasks.list(project=project_id)
        task_list = list(tasks)
        print(f"Found {len(task_list)} tasks to update")

        prediction_result = create_prediction_data(known_labels)

        if not prediction_result:
            print("No valid labels provided. Exiting.")
            return

        print("Prediction data to be applied:")
        for pred in prediction_result:
            print(f"  {pred['from_name']}: {pred['value']['choices'][0]}")

        success_count = 0
        error_count = 0

        for i, task in enumerate(task_list, 1):
            try:
                print(f"Updating task {i}/{len(task_list)} (ID: {task.id})...")

                prediction_data = {
                    "result": prediction_result,
                    "score": 1.0,
                    "model_version": "bulk_import_v1",
                }

                ls.predictions.create(
                    task=task.id,
                    result=prediction_data["result"],
                    score=prediction_data["score"],
                    model_version=prediction_data["model_version"],
                )

                success_count += 1
                print(f"  ✓ Successfully updated task {task.id}")

            except Exception as e:
                error_count += 1
                print(f"  ✗ Error updating task {task.id}: {str(e)}")

        print(f"\nBulk update completed!")
        print(f"Successfully updated: {success_count} tasks")
        print(f"Errors: {error_count} tasks")

    except Exception as e:
        print(f"Error fetching tasks: {str(e)}")


def main():
    known_labels = {
        "fuel_type": fuel_type,
        "steer_wheel": steer_wheel,
        "year": year,
        "color": color,
        "model": model,
        "make": make,
    }

    projects = ls.projects.list()

    for project in projects:
        project_id = project.id
        print(f"Project ID: {project_id}")
        print(f"Project Title: {project.title}")

        response = input(
            f"Do you want to update all tasks in project '{project.title}' (ID: {project_id}) with the known labels? (y/n): "
        )

        if response.lower() == "y":
            bulk_update_tasks(project_id, known_labels)
        else:
            print("Skipping this project.")

        print("-" * 50)


if __name__ == "__main__":
    main()
