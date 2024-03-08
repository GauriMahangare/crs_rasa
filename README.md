**Dependancies**
  - Models for recommendations - Content filtering, matrix factorisation and Deep neural network.
  - Mongo DB for tracker store
  - Django web project for chat bot and user profile.

**Steps**
1. Clone the Repository
2. Navigate to the Project Directory: After cloning the repository, navigate into the project directory using the cd command:
3. Create and Activate a Virtual Environment => python -m venv env
4. Activate the virtual environment => source bin/activate
5. Install Rasa => pip install rasa
6. Install Additional Dependencies => pip install -r requirements.txt
7. Train the Rasa Model: => rasa train
8. Run Rasa Server: => rasa run --log-file rasa.log --enable-api --cors="*"
9. To the run the action server rasa run actions
