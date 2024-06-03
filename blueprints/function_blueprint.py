from typing import List, Optional
from pydantic import BaseModel
from schemas import OpenAIChatMessage
import os
import requests
import json

from utils.pipelines.main import (
    get_last_user_message,
    add_or_update_system_message,
    get_tools_specs,
)


class Pipeline:
    class Valves(BaseModel):
        # List target pipeline ids (models) that this filter will be connected to.
        # If you want to connect this filter to all pipelines, you can set pipelines to ["*"]
        pipelines: List[str] = []

        # Assign a priority level to the filter pipeline.
        # The priority level determines the order in which the filter pipelines are executed.
        # The lower the number, the higher the priority.
        priority: int = 0

        # Valves for function calling
        OLLAMA_API_BASE_URL: str
        OLLAMA_API_KEY: str
        TASK_MODEL: str
        TEMPLATE: str

    def __init__(self):
        # Pipeline filters are only compatible with Open WebUI
        # You can think of filter pipeline as a middleware that can be used to edit the form data before it is sent to the OpenAI API.
        self.type = "filter"

        # Optionally, you can set the id and name of the pipeline.
        # Best practice is to not specify the id so that it can be automatically inferred from the filename, so that users can install multiple versions of the same pipeline.
        # The identifier must be unique across all pipelines.
        # The identifier must be an alphanumeric string that can include underscores or hyphens. It cannot contain spaces, special characters, slashes, or backslashes.
        # self.id = "function_calling_blueprint"
        self.name = "Function Calling Blueprint"

        # Initialize valves
        self.valves = self.Valves(
            **{
                "pipelines": ["*"],  # Connect to all pipelines
                "OLLAMA_API_BASE_URL": os.getenv(
                    "OLLAMA_API_BASE_URL", "http://localhost:11434"
                ),
                "OLLAMA_API_KEY": os.getenv("OLLAMA_API_KEY", "YOUR_OLLAMA_API_KEY"),
                "TASK_MODEL": os.getenv("TASK_MODEL", "llama3"),
                "TEMPLATE": """Use the following context as your learned knowledge, inside <context></context> XML tags.
<context>
    {{CONTEXT}}
</context>

When answer to user:
- If you don't know, just say that you don't know.
- If you don't know when you are not sure, ask for clarification.
Avoid mentioning that you obtained the information from the context.
And answer according to the language of the user's question.""",
            }
        )

    async def on_startup(self):
        # This function is called when the server is started.
        print(f"on_startup:{__name__}")
        pass

    async def on_shutdown(self):
        # This function is called when the server is stopped.
        print(f"on_shutdown:{__name__}")
        pass

    async def inlet(self, body: dict, user: Optional[dict] = None) -> dict:
        # If title generation is requested, skip the function calling filter
        if body.get("title", False):
            return body

        print(f"pipe:{__name__}")
        print(user)

        # Get the last user message
        user_message = get_last_user_message(body["messages"])

        # Get the tools specs
        tools_specs = get_tools_specs(self.tools)

        # System prompt for function calling
        fc_system_prompt = (
            f"Tools: {json.dumps(tools_specs, indent=2)}"
            + """
If a function tool doesn't match the query, return an empty string. Else, pick a function tool, fill in the parameters from the function tool's schema, and return it in the format { "name": \"functionName\", "parameters": { "key": "value" } }. Only pick a function if the user asks.  Only return the object. Do not return any other text."
"""
        )

        r = None
        try:
            # Costruzione dei messaggi per la richiesta
            messages = [
                {
                    "role": "system",
                    "content": fc_system_prompt,
                },
                {
                    "role": "user",
                    "content": "History:\n"
                    + "\n".join(
                        [
                            f"{message['role']}: {message['content']}"
                            for message in body["messages"][::-1][:4]
                        ]
                    )
                    + f"\nQuery: {user_message}",
                },
            ]
            # Call the OpenAI API to get the function response
            r = requests.post(
                url=f"{self.valves.OLLAMA_API_BASE_URL}/api/chat",
                json={
                    "model": self.valves.TASK_MODEL,
                    "messages": messages,
                    "stream": False
                },
                headers={
                    "Authorization": f"Bearer {self.valves.OLLAMA_API_KEY}",
                    "Content-Type": "application/json",
                },
            )
            r.raise_for_status()

            # Print the response text for debugging
            print("Response Text:", r.text)

            response = r.json()
            content = response["message"]["content"]

            # Parse the function response
            if content != "":
                result = json.loads(content)
                print(result)

                # Call the function
                if "name" in result:
                    function = getattr(self.tools, result["name"])
                    function_result = None
                    try:
                        function_result = function(**result["parameters"])
                    except Exception as e:
                        print(e)

                    # Add the function result to the system prompt
                    if function_result:
                        system_prompt = self.valves.TEMPLATE.replace(
                            "{{CONTEXT}}", function_result
                        )

                        print(system_prompt)
                        messages = add_or_update_system_message(
                            system_prompt, body["messages"]
                        )

                        # Return the updated messages
                        return {**body, "messages": messages}

        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e} - Response Text: {r.text if r else 'No response'}")
        except Exception as e:
            print(f"Error: {e}")
        
            if r:
                try:
                    print(r.json())
                except json.JSONDecodeError as e:
                    print(f"JSON decode error: {e} - Response Text: {r.text}")
                except Exception as e:
                    print(f"Failed to decode JSON and encountered an error: {e}")

        return body
