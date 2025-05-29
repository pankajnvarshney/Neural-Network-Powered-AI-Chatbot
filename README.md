# Neural-Network-Powered-AI-Chatbot

🤖 AI Chatbot using Deep Learning and NLP

This is a AI-based chatbot built using Python, deep learning, and natural language processing (NLP). The chatbot can understand user queries based on predefined intents and respond accordingly. It uses a feed-forward neural network (built with Keras) trained on labeled conversational data to classify intents and generate appropriate responses.


📌 Features

Intent classification using a neural network

Natural language processing using CountVectorizer and LabelEncoder

Easy-to-update conversation data (intents.json)

Streamlit-based user interface for interaction

Pretrained model (chatbot_model.keras) for fast deployment


🛠️ Technologies Used

Programming Language: Python

Libraries/Frameworks:

TensorFlow/Keras for building and training the neural network

Scikit-learn for preprocessing (CountVectorizer, LabelEncoder)

NumPy for data manipulation

Streamlit for creating the web app interface

Joblib for saving and loading preprocessing tools

🧠 Deep Learning

This project uses a simple Sequential neural network with dense layers for intent classification. Input text is vectorized and mapped to an intent class using softmax activation. The model is trained on a small set of example sentences for each intent.

📁 Project Structure

chatbot_project/

├── app.py                  # Main app file to run chatbot with Streamlit

├── train_model.py          # Script to train the intent classification model

├── chatbot_model.keras     # Trained deep learning model

├── intents.json            # Intents and responses in JSON format

├── vectorizer.pkl          # Pickled CountVectorizer object

├── label_encoder.pkl       # Pickled LabelEncoder object


🚀 How to Run

Clone the repository:

git clone https://github.com/your-username/ai-chatbot.git
cd ai-chatbot

Install the required packages:

pip install -r requirements.txt

To train the model:

python train_model.py


 Google Colab Setup Instructions
 
Run the following steps inside a Google Colab notebook to set up and launch your chatbot:

📁 Step 1: Upload Your Zip File

```python
from google.colab import files
uploaded = files.upload()  # Upload ai_chatbot_colab_ngrok.zip
```

📦 Step 2: Unzip the Project

```
!unzip -o ai_chatbot_colab_ngrok.zip -d chatbot_project
%cd chatbot_project
```

🛠️ Step 3: Install Required Packages

```
!pip install streamlit pyngrok keras scikit-learn joblib
```

🧠 Step 4: Train the Model

```python
!python train_model.py
```

🌐 Step 5: Launch Streamlit App with ngrok

```python
from pyngrok import ngrok
import os

!killall ngrok  # Terminate previous tunnels if running

ngrok.set_auth_token("YOUR_NGROK_AUTH_TOKEN")  # Replace with your actual ngrok token

os.system('streamlit run app.py &')
public_url = ngrok.connect(8501)
print("🔗 Open your chatbot here:", public_url)
```


![Screenshot (31)](https://github.com/user-attachments/assets/438c6c82-0f18-431a-8c70-cc293c636872)






Conclusion:

This project demonstrates how to build and deploy a simple yet functional AI chatbot using minimal tools and code. With basic NLP concepts, neural networks, and a clean Streamlit interface, it serves as a solid foundation for more advanced chatbot applications. The use of Colab and ngrok also shows how easy it is to deploy prototypes without needing a dedicated server.

