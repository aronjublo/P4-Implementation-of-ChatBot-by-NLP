import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import load_model

class ChatbotInterface:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        
        # Load the trained model and necessary files
        self.model = load_model('chatbot_model.h5')
        self.intents = json.loads(open('intent.json').read())
        self.words = pickle.load(open('words.pkl', 'rb'))
        self.classes = pickle.load(open('classes.pkl', 'rb'))

    def clean_up_sentence(self, sentence):
        sentence_words = nltk.word_tokenize(sentence)
        sentence_words = [self.lemmatizer.lemmatize(word.lower()) for word in sentence_words]
        return sentence_words

    def bag_of_words(self, sentence):
        sentence_words = self.clean_up_sentence(sentence)
        bag = [0] * len(self.words)
        for w in sentence_words:
            for i, word in enumerate(self.words):
                if word == w:
                    bag[i] = 1
        return np.array(bag)

    def predict_class(self, sentence):
        bow = self.bag_of_words(sentence)
        res = self.model.predict(np.array([bow]))[0]
        ERROR_THRESHOLD = 0.25
        results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
        results.sort(key=lambda x: x[1], reverse=True)
        return_list = []
        for r in results:
            return_list.append({'intent': self.classes[r[0]], 'probability': str(r[1])})
        return return_list

    def get_response(self, message):
        """Get chatbot response for user message"""
        ints = self.predict_class(message)
        
        # If no intent is matched with high confidence
        if not ints:
            # Check for numbers in the message for rank queries
            numbers = [int(s) for s in message.split() if s.isdigit()]
            if numbers and 1 <= numbers[0] <= 250:
                rank = numbers[0]
                # Find the movie intent for this rank
                for intent in self.intents['intents']:
                    if f"movie_{rank}" in str(intent['patterns']):
                        return random.choice(intent['responses'])
            return "I can help you with movie information! Try asking about a specific movie or rank (1-250)."
        
        tag = ints[0]['intent']
        list_of_intents = self.intents['intents']
        
        # Handle different types of intents
        for i in list_of_intents:
            if i['tag'] == tag:
                # Special handling for general movie queries
                if tag == "general_movie":
                    return random.choice(i['responses'])
                # Special handling for movie recommendations
                elif tag == "recommendations":
                    return random.choice(i['responses'])
                # Special handling for specific movie queries
                elif tag.startswith('movie_'):
                    return random.choice(i['responses'])
                # Handle all other intents (greetings, thanks, etc.)
                else:
                    return random.choice(i['responses'])
        
        return "I can tell you about any movie in the IMDB Top 250. Try asking about a specific movie or rank!"

# Training code (only runs if file is executed directly)
if __name__ == "__main__":
    print("Initializing training...")
    
    lemmatizer = WordNetLemmatizer()
    intents = json.loads(open('intent.json').read())

    words = []
    classes = []
    documents = []
    ignore_letters = ['?', '!', '.', ',']

    for intent in intents['intents']:
        for pattern in intent['patterns']:
            word_list = nltk.word_tokenize(pattern)
            words.extend(word_list)
            documents.append((word_list, intent['tag']))
            if intent['tag'] not in classes:
                classes.append(intent['tag'])

    words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore_letters]
    words = sorted(set(words))
    classes = sorted(set(classes))

    print(f"Unique words: {len(words)}")
    print(f"Classes: {len(classes)}")
    print(f"Documents: {len(documents)}")

    pickle.dump(words, open('words.pkl', 'wb'))
    pickle.dump(classes, open('classes.pkl', 'wb'))

    training = []
    output_empty = [0] * len(classes)

    for document in documents:
        bag = []
        word_patterns = document[0]
        word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
        for word in words:
            bag.append(1) if word in word_patterns else bag.append(0)

        output_row = list(output_empty)
        output_row[classes.index(document[1])] = 1
        training.append([bag, output_row])

    random.shuffle(training)
    training = np.array(training, dtype=object)

    train_x = list(training[:, 0])
    train_y = list(training[:, 1])

    print("Building neural network model...")
    model = Sequential()
    model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(len(train_y[0]), activation='softmax'))

    sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    print("Training model...")
    hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
    print(f"Model trained with final accuracy: {hist.history['accuracy'][-1]:.4f}")

    model.save('chatbot_model.h5')
    print("Model saved as 'chatbot_model.h5'")
    print("Training completed successfully!") 