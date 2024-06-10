import os
import requests
from typing import Literal, List, Optional
from datetime import datetime
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk, json
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

from blueprints.function_blueprint import Pipeline as FunctionCallingBlueprint

def web_scraper(url):
    response = requests.get("https://r.jina.ai/" + url)
    tokens = word_tokenize(response.text)
    tokens = [token for token in tokens if token.isalnum()]  # Remove non-alphanumeric tokens
    tokens = [token for token in tokens if token not in stopwords.words('english')]  # Remove stopwords
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]  # Lemmatize tokens
    formatted_text = ' '.join(tokens)
    sentences = sent_tokenize(formatted_text)
    return sentences

def format_search_question(query):
    now = datetime.datetime.today()
    formatted_now = now.strftime("%Y-%m-%d")
    api_key = os.getenv('AI_API_Key')
    URL = os.getenv('AI_URL')
    URL = URL + "/generate"
    payload = {
        "model": os.getenv('AI_MODEL'),  # You can adjust this value to control the response tone (0-9)
        "prompt": f"You have this question from user: '{query}'. Rephrase the question for a better web search using a MAXIMUM of 5 words, do not use more than 5 words. Reply only with the rephrased question.",
        "stream": False
    }
    # Set the API key in the request headers
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    # Make the POST request to the OLLAMA API
    response = requests.post(URL, json=payload, headers=headers)
    # Check if the request was successful
    if response.status_code == 200:
        # Get the response text from the API
        response_text = response.json()["response"]
        # Send the response back to the user
        return response_text

class Pipeline(FunctionCallingBlueprint):
    class Valves(FunctionCallingBlueprint.Valves):
        # Add your custom parameters here
        OPENWEATHERMAP_API_KEY: str = ""
        BRAVE_API_KEY: str = ""
        pass

    class Tools:
        def __init__(self, pipeline) -> None:
            self.pipeline = pipeline

        def get_current_time(
            self,
        ) -> str:
            """
            Get the current time.

            :return: The current time.
            """

            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            return f"Current Time = {current_time}"

        def get_current_date(
            self,
        ) -> str:
            """
            Get the current date.

            :return: The current date.
            """

            now = datetime.now()
            current_date = now.strftime("%A, %B %d, %Y")
            return f"Current Date = {current_date}"

        def get_current_weather(
            self,
            location: str,
            unit: Literal["metric", "fahrenheit"] = "fahrenheit",
        ) -> str:
            """
            Get the current weather for a location. If the location is not found, return an empty string.

            :param location: The location to get the weather for.
            :param unit: The unit to get the weather in. Default is fahrenheit.
            :return: The current weather for the location.
            """

            # https://openweathermap.org/api

            if self.pipeline.valves.OPENWEATHERMAP_API_KEY == "":
                return "OpenWeatherMap API Key not set, ask the user to set it up."
            else:
                units = "imperial" if unit == "fahrenheit" else "metric"
                params = {
                    "q": location,
                    "appid": self.pipeline.valves.OPENWEATHERMAP_API_KEY,
                    "units": units,
                }

                response = requests.get(
                    "http://api.openweathermap.org/data/2.5/weather", params=params
                )
                response.raise_for_status()  # Raises an HTTPError for bad responses
                data = response.json()

                weather_description = data["weather"][0]["description"]
                temperature = data["main"]["temp"]

                return f"{location}: {weather_description.capitalize()}, {temperature}°{unit.capitalize()[0]}"


        def bravesearch(
                self,
                query: str,
        ) -> str:
            """
            Perform a web search with Brave search and scrape the websites founded in the search.

            :param query: the user question.
            :return: Web search results.
            """
            if self.pipeline.valves.BRAVE_API_KEY == "":
                return "OpenWeatherMap API Key not set, ask the user to set it up."
            else:
                current_date = datetime.date.today()
                month_ago = current_date - datetime.timedelta(days=60)
                api_key = self.pipeline.valves.BRAVE_API_KEY
                url = 'https://api.search.brave.com/res/v1/web/search'
                # Set query parameters
                params = {
                    "q": query,
                    "Accept": "application/json"
                }
                # Add API key to headers
                headers = {
                    "X-Subscription-Token": api_key,
                }
                # Make GET request with compressed response
                response = requests.get(url, params=params, headers=headers, stream=True)
                data = json.loads(response.text)
                #print(data)
                try:
                    results = data["web"]["results"]
                except:
                    check = True
                    while check:
                        print("Wrong format in the response, retry the query")
                        regen_query = format_search_question(query)
                        print(regen_query)
                        params_new = {
                            "q": regen_query,
                            "Accept": "application/json"
                        }
                        response = requests.get(url, params=params_new, headers=headers, stream=True)
                        data = json.loads(response.text)
                        #print(data)
                        try:
                            results = data["web"]["results"]
                            check = False
                        except KeyError as e:
                            print(f"Error: {e}")
                href_values = []
                for result in results:
                    if "url" in result and "page_age" in result and datetime.datetime.strptime(result["page_age"], "%Y-%m-%dT%H:%M:%S").date() > month_ago:
                        href_values.append(result["url"])

                contents = []
                print(href_values)
                for href in href_values:
                    print("Scanning: ",href)
                    try:
                        contents = web_scraper(href)
                    except Exception as e:
                        print (f"connection error: {e}")

                return contents

    def __init__(self):
        super().__init__()
        # Optionally, you can set the id and name of the pipeline.
        # Best practice is to not specify the id so that it can be automatically inferred from the filename, so that users can install multiple versions of the same pipeline.
        # The identifier must be unique across all pipelines.
        # The identifier must be an alphanumeric string that can include underscores or hyphens. It cannot contain spaces, special characters, slashes, or backslashes.
        # self.id = "my_tools_pipeline"
        self.name = "My Tools Pipeline"
        self.valves = self.Valves(
            **{
                **self.valves.model_dump(),
                "pipelines": ["*"],  # Connect to all pipelines
                "OPENWEATHERMAP_API_KEY": os.getenv("OPENWEATHERMAP_API_KEY", ""),
                "BRAVE_API_KEY": os.getenv("BRAVE_API_KEY", ""),
            },
        )
        self.tools = self.Tools(self)
