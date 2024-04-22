#%%
import math
from matplotlib import gridspec
import matplotlib.pyplot as plt
import numpy as np
import urllib.request
from PIL import Image
import io
import re
import copy
import os
import cv2
import base64
from io import BytesIO
import requests
import openai
from tenacity import retry, stop_after_attempt, wait_fixed
from IPython.display import Markdown,display
from rich.console import Console
import json
import os
import sys
sys.path.append(f'{os.path.dirname(os.path.abspath(__file__))}/../utils')
from util_visualize import show_images, visualize_subplots

#%%
def set_openai_api_key_from_txt(key_path='./key.txt',VERBOSE=True):
    """
        Set OpenAI API Key from a txt file
    """
    with open(key_path, 'r') as f: 
        OPENAI_API_KEY = f.read()
    openai.api_key = OPENAI_API_KEY
    if VERBOSE:
        print ("OpenAI API Key Ready from [%s]."%(key_path))
    
class GPTchatClass():
    def __init__(self,
                 gpt_model = 'gpt-4',
                 role_msg  = 'Your are a helpful assistant.',
                 VERBOSE   = True
                ):
        self.gpt_model     = gpt_model
        self.messages      = [{'role':'system','content':f'{role_msg}'}]
        self.init_messages = [{'role':'system','content':f'{role_msg}'}]
        self.VERBOSE       = VERBOSE
        self.response      = None
        self.detail        = "auto"

        if self.VERBOSE:
            print ("Chat agent using [%s] initialized with the follow role:[%s]"%
                   (self.gpt_model,role_msg))
    
    def _add_message(self,role='assistant',content=''):
        """
            role: 'assistant' / 'user'
        """
        self.messages.append({'role':role, 'content':content})
        
    def _get_response_content(self):
        if self.response:
            return self.response['choices'][0]['message']['content']
        else:
            return None
        
    def _get_response_status(self):
        if self.response:
            return self.response['choices'][0]['message']['finish_reason']
        else:
            return None
    
    @retry(stop=stop_after_attempt(10), wait=wait_fixed(5))
    def chat(self,user_msg='hi',
             PRINT_USER_MSG=True,PRINT_GPT_OUTPUT=True,
             RESET_CHAT=False,RETURN_RESPONSE=True):
        self._add_message(role='user',content=user_msg)
        self.response = openai.ChatCompletion.create(
            model    = self.gpt_model,
            messages = self.messages
        )
        # Backup response for continous chatting
        self._add_message(role='assistant',content=self._get_response_content())
        if PRINT_USER_MSG:
            print("[USER_MSG]")
            printmd(user_msg)
        if PRINT_GPT_OUTPUT:
            print("[GPT_OUTPUT]")
            printmd(self._get_response_content())
        # Reset
        if RESET_CHAT:
            self.messages =  copy.copy(self.init_messages)
        # Return
        if RETURN_RESPONSE:
            return self._get_response_content()

class GPT4VisionClass:
    def __init__(
            self, 
            gpt_model: str = "gpt-4-vision-preview",
            role_msg: str = "You are a helpful agent with vision capabilities; do not respond to objects not depicted in images.",
            key_path='../key/rilab_key.txt', 
            max_tokens = 512, temperature = 0.9, n = 1, stop = [], VERBOSE=True,
            image_max_size:int = 512,
            ):
        self.gpt_model = gpt_model
        self.role_msg = role_msg
        self.messages = [{"role": "system", "content": f"{role_msg}"}]
        self.init_messages = [{"role": "system", "content": f"{role_msg}"}]
        self.history = [{"role": "system", "content": f"{role_msg}"}]
        self.image_max_size = image_max_size

        # GPT-4 parameters
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.n = n
        self.stop = stop
        self.VERBOSE = VERBOSE
        if self.VERBOSE:
            self.console = Console()
        self.response = None
        self.image_token_count = 0

        self._setup_client(key_path)

    def _setup_client(self, key_path):
        if self.VERBOSE:
            self.console.print(f"[bold cyan]key_path:[%s][/bold cyan]" % (key_path))

        with open(key_path, "r") as f:
            OPENAI_API_KEY = f.read()
        self.client = openai.OpenAI(api_key=OPENAI_API_KEY)

        if self.VERBOSE:
            self.console.print(
                "[bold cyan]Chat agent using [%s] initialized with the follow role:[%s][/bold cyan]"
                % (self.gpt_model, self.role_msg)
            )
    
    def _backup_chat(self):
        self.init_messages = copy.copy(self.messages)

    def _get_response_content(self):
        if self.response:
            return self.response.choices[0].message.content
        else:
            return None

    def _get_response_status(self):
        if self.response:
            return self.response.choices[0].message.finish_reason
        else:
            return None
        
    def _encode_image_path(self, image_path):
        # with open(image_path, "rb") as image_file:
        image_pil = Image.open(image_path)
        image_pil.thumbnail(
            (self.image_max_size, self.image_max_size)
        )
        image_pil_rgb = image_pil.convert("RGB")
        # change pil to base64 string
        img_buf = io.BytesIO()
        image_pil_rgb.save(img_buf, format="PNG")
        return base64.b64encode(img_buf.getvalue()).decode('utf-8')

    def _encode_image(self, image):
        """
            Save the image to a temporary file and encode it to base64
        """
        # save Image:PIL to temp file
        cv2.imwrite("temp.jpg", np.array(image))
        with open("temp.jpg", "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
        os.remove("temp.jpg")
        return encoded_image

    def _count_image_tokens(self, width, height):
        h = ceil(height / 512)
        w = ceil(width / 512)
        n = w * h
        total = 85 + 170 * n
        return total

    def set_common_prompt(self, common_prompt):
        self.messages.append({"role": "system", "content": common_prompt})

    # @retry(stop=stop_after_attempt(10), wait=wait_fixed(5))
    def chat(
            self, 
            query_text, 
            image_paths=[], images=None, APPEND=True,
            PRINT_USER_MSG=True,
            PRINT_GPT_OUTPUT=True,
            RESET_CHAT=False,
            RETURN_RESPONSE=True,
            MAX_TOKENS = 512,
            VISUALIZE = False,
            DETAIL = "auto",
            CROP = None,
            ):
        """
            image_paths: list of image paths
            images: list of images
            You can only provide either image_paths or image.
        """
        if DETAIL:
            self.console.print(f"[bold cyan]DETAIL:[/bold cyan] {DETAIL}")
            self.detail = DETAIL
        content = [{"type": "text", "text": query_text}]
        content_image_not_encoded = [{"type": "text", "text": query_text}]
        # Prepare the history temp
        if image_paths is not None:
            local_imgs = []
            for image_path_idx, image_path in enumerate(image_paths):
                with Image.open(image_path) as img:
                    width, height = img.size
                    if CROP:
                        img = img.crop(CROP)
                        width, height = img.size
                        # convert PIL to numpy array
                        local_imgs.append(np.array(img))
                    self.image_token_count += self._count_image_tokens(width, height)

                print(f"[{image_path_idx}/{len(image_paths)}] image_path: {image_path}")
                base64_image = self._encode_image_path(image_path)
                image_content = {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                            "detail": self.detail
                        }
                    }
                image_content_in_numpy_array = {
                        "type": "image_numpy",
                        "image": np.array(Image.open(image_path))
                    }
                content.append(image_content)
                content_image_not_encoded.append(image_content_in_numpy_array)
        elif images is not None:
            local_imgs = []
            for image_idx, image in enumerate(images):
                image_pil = Image.fromarray(image)
                if CROP:
                    image_pil = image_pil.crop(CROP)
                    local_imgs.append(image_pil)
                    # width, height = image_pil.size
                image_pil.thumbnail(
                    (self.image_max_size, self.image_max_size)
                )
                width, height = image_pil.size
                self.image_token_count += self._count_image_tokens(width, height)
                self.console.print(f"[deep_sky_blue3][{image_idx+1}/{len(images)}] Image provided: [Original]: {image.shape}, [Downsize]: {image_pil.size}[/deep_sky_blue3]")
                base64_image = self._encode_image(image_pil)
                image_content = {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                            "detail": self.detail
                        }
                    }
                image_content_in_numpy_array = {
                        "type": "image_numpy",
                        "image": image
                    }
                content.append(image_content)
                content_image_not_encoded.append(image_content_in_numpy_array)
        else:
            self.console.print("[bold red]Neither image_paths nor images are provided.[/bold red]")

        if VISUALIZE:
            if image_paths:
                self.console.print("[deep_sky_blue3][VISUALIZE][/deep_sky_blue3]")
                if CROP:
                    visualize_subplots(local_imgs)
                else:
                    visualize_subplots(image_paths)
            elif images:
                self.console.print("[deep_sky_blue3][VISUALIZE][/deep_sky_blue3]")
                if CROP:
                    local_imgs = np.array(local_imgs)
                    visualize_subplots(local_imgs)
                else:
                    visualize_subplots(images)

        self.messages.append({"role": "user", "content": content})
        self.history.append({"role": "user", "content": content_image_not_encoded})
        payload = self.create_payload(model=self.gpt_model)
        self.response = self.client.chat.completions.create(**payload)

        if PRINT_USER_MSG:
            self.console.print("[deep_sky_blue3][USER_MSG][/deep_sky_blue3]")
            print(query_text)
        if PRINT_GPT_OUTPUT:
            self.console.print("[spring_green4][GPT_OUTPUT][/spring_green4]")
            print(self._get_response_content())
        # Reset
        if RESET_CHAT:
            self.messages = self.init_messages
        # Return
        if RETURN_RESPONSE:
            return self._get_response_content()

    @retry(stop=stop_after_attempt(10), wait=wait_fixed(5))
    def chat_multiple_images(self, image_paths, query_text, model="gpt-4-vision-preview", max_tokens=300):
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": query_text}]
            }
        ]
        for image_path in image_paths:
            base64_image = self._encode_image(image_path)
            messages[0]["content"].append(
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            )
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens
        )
        return response

    @retry(stop=stop_after_attempt(10), wait=wait_fixed(5))
    def generate_image(self, prompt, size="1024x1024", quality="standard", n=1):
        response = self.client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size=size,
            quality=quality,
            n=n
        )
        return response

    def visualize_image(self, image_response):
        image_url = image_response.data[0].url
        # Open the URL and convert the image to a NumPy array
        with urllib.request.urlopen(image_url) as url:
            img = Image.open(url)
            img_array = np.array(img)

        plt.imshow(img_array)
        plt.axis('off')
        plt.show()

    def create_payload(self,model):
        payload = {
            "model": model,
            "messages": self.messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "n": self.n
        }
        if len(self.stop) > 0:
            payload["stop"] = self.stop
        return payload
    
    def save_interaction(self, data, file_path: str = "./scripts/interaction_history.json"):
        """
        Save the chat history to a JSON file.
        The history includes the user role, content, and images stored as NumPy arrays.
        """
        self.history = data.copy()
        history_to_save = []
        for entry in self.history:
            entry_to_save = {
                "role": entry["role"],
                "content": []
            }
            # Check if 'content' is a string or a list
            if isinstance(entry["content"], str):
                entry_to_save["content"].append({"type": "text", "text": entry["content"]})
            elif isinstance(entry["content"], list):
                for content in entry["content"]:
                    if content["type"] == "text":
                        entry_to_save["content"].append(content)
                    elif content["type"] == "image_numpy":
                        entry_to_save["content"].append({"type": "image_numpy", "image": content["image"].tolist()})
                    elif content["type"] == "image_url":
                        entry_to_save["content"].append(content)
            history_to_save.append(entry_to_save)
        with open(file_path, "w") as file:
            json.dump(history_to_save, file, indent=4)

        if self.VERBOSE:
            self.console.print(f"[bold green]Chat history saved to {file_path}[/bold green]")

    def get_total_token(self):
        """
            Get total token used
        """
        if self.VERBOSE:
            self.console.print(f"[bold cyan]Total token used: {self.response.usage.total_tokens}[/bold cyan]")
        return self.response.usage.total_tokens
    
    def get_image_token(self):
        """
            Get image token used
        """
        if self.VERBOSE:
            self.console.print(f"[bold cyan]Image token used: {self.image_token_count}[/bold cyan]")
        return self.image_token_count

    def reset_tokens(self):
        """
            Reset total and image token used
        """
        self.response.usage.total_tokens = 0
        self.image_token_count = 0
        if self.VERBOSE:
            self.console.print(f"[bold cyan]Image token reset[/bold cyan]")

from math import ceil

def count_image_tokens(width: int, height: int):
    h = ceil(height / 512)
    w = ceil(width / 512)
    n = w * h
    total = 85 + 170 * n
    return total

def printmd(string):
    display(Markdown(string))
    
def extract_quoted_words(string):
    quoted_words = re.findall(r'"([^"]*)"', string)
    return quoted_words

def response_to_json(response):
    # Remove the markdown code block formatting
    response_strip = response.strip('```json\n').rstrip('```')
    # Convert the cleaned string to a JSON object
    try:
        response_json = json.loads(response_strip)
    except json.JSONDecodeError as e:
        response_json = None
        error_message = str(e)

    return response_json, error_message if response_json is None else ""

def match_objects(response_object_names, original_object_names, type_conversion):
    matched_objects = []

    for res_obj_name in response_object_names:
        components = res_obj_name.split('_')
        converted_components = set()

        # Applying type conversion and creating a unique set of components
        for comp in components:
            converted_comp = type_conversion.get(comp, comp)
            converted_components.add(converted_comp)
        # Check if the unique set of converted components is in any of the original object names
        for original in original_object_names:
            if all(converted_comp in original for converted_comp in converted_components):
                matched_objects.append(original)
                break
        else:
            print(f"No match found for {res_obj_name}")
            print(f"Type manually in the set of {original_object_names}:")
            matched_objects.append(input())

    return matched_objects

def parse_and_get_action(response_json, option_idx, original_objects, type_conversion):
    func_call_list = []
    action = response_json["options"][option_idx-1]["action"]

    # Splitting actions correctly if there are multiple actions
    if isinstance(action, str):
        actions = [act.strip() + ')' for act in action.split('),') if act.strip()]
    elif isinstance(action, list):
        actions = action
    else:
        raise ValueError("Action must be a string or a list of strings")

    for act in actions:
        # Handle special cases; none-action / done-action
        if act in ["move_object(None, None)", "set_done()"]:
            func_call_list.append(f"{act}")
            continue

        # Regular action processing
        func_name, args = act.split('(', 1)
        args = args.rstrip(')')
        args_list = args.split(', ')
        new_args = []

        for arg in args_list:
            arg_parts = arg.split('_')
            # Applying type conversion to each part of arg
            converted_arg_parts = [type_conversion.get(part, part) for part in arg_parts]
            matched_name = match_objects(["_".join(converted_arg_parts)], original_objects, type_conversion)
            if matched_name:
                arg = matched_name[0]

            new_args.append(f'"{arg}"')

        func_call = f"{func_name}({', '.join(new_args)})"
        func_call_list.append(func_call)

    return func_call_list

def parse_actions_to_executable_strings(response_json, option_idx, env):
    actions = response_json["options"][option_idx - 1]["actions"]
    executable_strings = []
    stored_results = {}

    for action in actions:
        function_name = action["function"]
        arguments = action["arguments"]

        # Preparing the arguments for the function call
        prepared_args = []
        for arg in arguments:

            if arg == "None":  # Handling the case where the argument is "None"
                prepared_args.append(None)
            elif arg in stored_results:
                # Use the variable name directly
                prepared_args.append(stored_results[arg])
            else:
                # Format the argument as a string or use as is
                prepared_arg = f'"{arg}"' if isinstance(arg, str) else arg
                prepared_args.append(prepared_arg)

        # Format the executable string
        if "store_result_as" in action:
            result_var = action["store_result_as"]
            exec_str = f'{result_var} = env.{function_name}({", ".join(map(str, prepared_args))})'
            stored_results[result_var] = result_var  # Store the variable name for later use
        else:
            exec_str = f'env.{function_name}({", ".join(map(str, prepared_args))})'

        executable_strings.append(exec_str)

    return executable_strings

def extract_arguments(response_json):
    # Regular expression pattern to extract arguments from action
    pattern = r'move_object\(([^)]+)\)'

    # List to hold extracted arguments
    extracted_arguments = []

    # Iterate over each option in response_json
    for option in response_json.get("options", []):
        action = option.get("action", "")
        match = re.search(pattern, action)

        if match:
            # Extract the content inside parentheses and split by comma
            arguments = match.group(1)
            args = [arg.strip() for arg in arguments.split(',')]
            extracted_arguments.append(args)

    return extracted_arguments

def decode_image(base64_image_string):
    """
    Decodes a Base64 encoded image string and returns it as a NumPy array.
    
    Parameters:
    base64_image_string (str): A Base64 encoded image string.
    
    Returns:
    numpy.ndarray: A NumPy array representing the image if successful, None otherwise.
    """
    # Remove Data URI scheme if present
    if "," in base64_image_string:
        base64_image_string = base64_image_string.split(',')[1]

    try:
        image_data = base64.b64decode(base64_image_string)
        image = Image.open(BytesIO(image_data))
        return np.array(image)
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
