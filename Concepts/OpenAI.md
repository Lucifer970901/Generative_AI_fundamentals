# Open AI
OpenAI  focuses on developing and deploying proprietary, state-of-the-art AI models through paid APIs. OpenAI is best known for creating advanced Generative AI models that can produce human-like text, images, and other forms of content based on simple prompts.

| Product/Model |	Core Function |	Main Use Cases |
|:--- | :--- |:--- |
| ChatGPT (GPT Series) |	Generative Pre-trained Transformer (Large Language Models) |	Conversation and Text Generation: Writing essays, answering complex questions, summarizing documents, creating code, customer support chatbots, and personalized learning. |
| DALL-E	Text-to-Image Generation |	Image Creation: Generating unique artwork, marketing visuals, product mockups, and designs from natural language descriptions.|
| Sora| 	Text-to-Video Generation |	Video Creation: Creating realistic and imaginative scenes and videos based on text prompts.
Whisper	Automatic Speech Recognition (ASR)	Audio Transcription: Transcribing speech from audio files into text with high accuracy, and translating audio across languages. |
| OpenAI API	| Developer Platform |	Integrating AI: Allows businesses and developers to integrate any of the core models (GPT, DALL-E, Whisper, etc.) into their own applications, workflows, and custom AI solutions. |

##  Token-Based Charges & Model Endpoints

* Charges Based on Token Size:

Pricing is usage-based, calculated on the number of Input Tokens (your prompt, system instructions, and history) and Output Tokens (the model's response).

This necessitates setting limits on response length to control costs.

* System Information:

A set of instructions provided to the Chat Completion API at the beginning of the conversation (role: system).

It defines the model's persona (e.g., "You are a helpful assistant"), constraints, and rules for the entire interaction.

* API Model Distinction:

    * Chat Completion API (/v1/chat/completions): Used for the latest and most capable models (e.g., gpt-4o, gpt-5-nano). Requires a structured message array (system,  user, assistant).

    *   Completion API (/v1/completions): A legacy endpoint used primarily for the older, simpler GPT Base Models (e.g., ada, davinci). It takes a single, unstructured text prompt.

## Model Control Parameters

**Temperature (Randomness Control)**:

Effect: Controls the randomness or "creativity" of the output by scaling the probability distribution of the next token.

Close to 0 (e.g., 0.1): Makes the model highly deterministic and predictable, consistently selecting the highest probability tokens. Ideal for factual extraction or technical code generation.

Higher values (e.g., 0.8): Flattens the distribution, allowing lower-probability tokens to be selected, resulting in more diverse and creative output.

**Top P (Nucleus Sampling)**:

Effect: Controls the diversity by defining a dynamic "nucleus" of tokens from which to sample.

The model only considers the tokens whose cumulative probability adds up to the value of top_p.

Low top_p (e.g., 0.1): Restricts the model to the most confident, highest-probability choices.

Note: It's generally recommended to adjust Temperature or Top P, but not both aggressively.

**Maximum Tokens**:

Purpose: Sets the upper limit for the number of Output Tokens the model is allowed to generate in its response.

Function: Serves as a primary cost control mechanism and ensures the response doesn't exceed the total context window size.

## Function Calling (Tool Use)

Mechanism: A feature where the model is provided with descriptions of external functions/tools (e.g., get_current_weather).

Process:

* The model analyzes the user's prompt.

* If an external action is required, the model responds with a structured JSON object containing the function name and necessary arguments.

* The application code executes the real function using those arguments.

* The application sends the function's output (e.g., the current weather data) back to the model.

* The model uses the output to generate a final, natural language answer for the user.

    Purpose: Enables the model to interact with external APIs and services, moving beyond text generation into reliable agentic behavior.

    Note: The parameter that specifies how many different responses you want to generate is called n, which is often set to 1.