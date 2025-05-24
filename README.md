🪑 Furninder — Find Similar Furniture Using AI
==============================================

Furninder is an AI-powered application that helps you find visually similar furniture items simply by uploading an image. Leveraging Google's **PaLI-Gemma** multimodal model, Furninder intelligently performs image-to-text conversion and then finds the closest matches based using clip embeddings.


✨ Features
----------

*   **Image-to-Image Search**: Find furniture that looks like the one in your photo.
    
*   **AI-Powered Similarity Matching**: Utilizes the advanced **PaLI-Gemma** model for accurate results.
    
*   **User-Friendly Interface**: Built with Flask for easy interaction.
    

🚀 Setup Instructions
---------------------

Get Furninder up and running with these simple steps:

### 1\. Prerequisites

*   **Python**: Ensure you have Python version **3.8 or newer** installed.
    
*   **Hugging Face Account**: You'll need a Hugging Face account and a generated access token for model authentication.
    

### 2\. Clone the Repository

Start by cloning the Furninder repository to your local machine. Open your terminal or command prompt and run:

git clone https://github.com/AbhinandIITM/Appian-hackathon-R2.git


### 3\. Install Dependencies

Navigate into the cloned directory and install the required Python packages. From your terminal, run:

pip install -r requirements.txt

### 4\. Hugging Face Authentication

If this is your first time using models from Hugging Face, you'll need to authenticate. This allows Furninder to download and use the PaLI-Gemma model. Run this command and follow the prompts:

huggingface-cli login

▶️ Running the Application
--------------------------

Once you've completed the setup, you're ready to launch Furninder!

1.  cd flask
    
2.  python app.py
    

**⚠️ Note**: The first time you run the application, it will **download the PaLI-Gemma model weights**. This is a large file and may take some time depending on your internet connection. Please be patient.

⚠️ Disclaimer
-------------

**Hardware Performance**: The performance of Furninder, particularly inference times, can vary significantly based on your hardware.

**Recommended System**: This model was developed and thoroughly tested on an **NVIDIA RTX 4060**. Systems with lower GPU resources may experience slower processing speeds.
