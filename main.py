import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from datetime import datetime
import joblib   # para salvar e carregar modelos

def load_data(path="data/raw/SMSSpamCollection"):
    data = pd.read_csv(path, sep='\t', header=None, names=["label", "text"])
    return data

def preprocess(texts, labels):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(texts)
    y = (labels == "spam").astype(int)
    return X, y, vectorizer

def train_model(X_train, y_train):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model

def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    cm = confusion_matrix(y_test, preds)
    return acc, cm

def log_results(acc, cm, dataset="SMS Spam Collection"):
    now = datetime.now().strftime("%d/%m/%Y %H:%M")
    with open("docs/notas.md", "a", encoding="utf-8") as f:
        f.write(f"\n## Execução – {now}\n")
        f.write(f"- Dataset: {dataset}\n")
        f.write(f"- Resultado: Acurácia {acc:.2f}\n")
        f.write(f"- Matriz de confusão:\n{cm}\n")
        f.write(f"- Observações: modelo treinado com Logistic Regression.\n")

def save_model(model, vectorizer, path_model="models/spam_model.pkl", path_vectorizer="models/vectorizer.pkl"):
    """
    Salva o modelo e o vetor de palavras para reuso.
    """
    joblib.dump(model, path_model)
    joblib.dump(vectorizer, path_vectorizer)

def load_model(path_model="models/spam_model.pkl", path_vectorizer="models/vectorizer.pkl"):
    """
    Carrega modelo e vetor previamente salvos.
    """
    model = joblib.load(path_model)
    vectorizer = joblib.load(path_vectorizer)
    return model, vectorizer

def main():
    data = load_data()
    X, y, vectorizer = preprocess(data["text"], data["label"])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = train_model(X_train, y_train)
    acc, cm = evaluate(model, X_test, y_test)

    print(f"Acurácia: {acc:.2f}")
    print("Matriz de confusão:")
    print(cm)

    # Documenta automaticamente
    log_results(acc, cm)

    # Salva modelo e vetor
    save_model(model, vectorizer)

if __name__ == "__main__":
    main()