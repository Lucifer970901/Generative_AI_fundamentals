
# Large Language Models (LLMs)

These are foundational machine learning models that usedeep learning algorithms to process and understand natural language, these models are trained on massive amount of text data to learn patterns and entity relationships in the language.

This is the type of generative AI models that specializes in undestanding, generating and interacting in human language. its responsible to perform tasks such as text to text generation, text to image generation and image to text generation.


### what makes LLMs so powerful?

In case of LLM, one model can be used for variety of tasks like text generation, chatbot, summarizer, translation code generation and so on.
its called ***multimodal*** as it can perform multiple tasks.

So LLM is subset of Deep Learning and it has properties that merge with Generative AI.

Some examples for LLM models include, gemini, GPT, XLM, T5, Llama, Mistral, Falcon etc.

## End to end Generative AI pipeline.
 
it consists of steps used to build end to end GenAI software.

Break the problem down into several sub-problems, then try to develop step by step procedure to solve them. Since language processing is involved we would also list all the forms of text processing needed in each step. This step by step processing of text is known as pipeline.

This includes
* Data aquisition
* Data prepraration
* Feature Engineering
* Modeling
* Evaluation
* Deployment
* Monitoring and model updating.

### Data aquisition

First, we need to assess our current data availability. We'll check for any existing data files in common formats such as CSV, DOCX, XLSX, PDF, or TXT.

1. **External Data Sources**: If internal files are insufficient, we will explore external data sources. This includes:

* Querying databases (DB).
* Searching the internet for publicly available datasets.
* Utilizing APIs to collect data programmatically.
* Implementing web scraping techniques.

2. **Data Generation and Augmentation**: In the event we find ***no existing data***, we will proceed to ***create our own dataset***, potentially leveraging tools like the OpenAI API for generation.

If we have ***limited data***, we will ***perform data augmentation*** to expand the dataset size and variety, ensuring we have enough information for robust analysis or model training.

3. **Textual Data Augmentation**

Textual augmentation methods modify the existing text to create diverse variations while preserving the original meaning:

* **Synonym Replacement**: Swap words with their ***synonyms*** (e.g., "I am AI engineer" becomes "I am artificial intelligence engineer").

    * **Bigram Flip**: ***Swap adjacent word pairs*** to introduce grammatical variations, helping models become more robust to incorrect syntax (e.g., "I am going to the supermarket" becomes "I am to going the supermarket").

    * **Back Translation**: Translate the text to one or more foreign languages and then ***translate it back*** to the original language. This generates a paraphrased, non-identical version of the original sentence. This technique is most effective with larger datasets.

    * **Adding Noise/Context**: Append a small, ***contextually relevant sentence or phrase to the original text*** (e.g., "I am a data scientist" becomes "I am a data scientist**, i love my job**").

4. **Image Data Augmentation**

Image augmentation applies transformations to images to increase the quantity and variability of the training data:

    * **Geometric Transformations**:

        * Flips: Horizontal flip, Vertical flip.
        * Rotation: Positive and negative rotation.
        * Crop: Random cropping.

    * **Color/Pixel Manipulations**:

        * Brightness and Contrast adjustments.
        * Color Space Conversion: Convert to Grayscale.
        * Blur and Noise addition.