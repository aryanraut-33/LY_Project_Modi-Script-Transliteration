import os
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from src.preprocessing import run_preprocessing
from src.model_factory import get_model
import config

def plot_history(history, method):
    """
    Saves accuracy and loss plots to verify the 'Drop-out Induced' behavior 
    as described in the research paper.
    """
    os.makedirs(config.LOG_DIR, exist_ok=True)
    
    # Save history to CSV for numerical analysis
    hist_df = pd.DataFrame(history.history)
    hist_df.to_csv(os.path.join(config.LOG_DIR, f"{method}_history.csv"), index=False)

    # Plot Accuracy and Loss
    plt.figure(figsize=(12, 5))
    
    # Accuracy Plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title(f'{method.upper()} - Classification Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title(f'{method.upper()} - Categorical Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(config.LOG_DIR, f"{method}_performance.png"))
    plt.close()
    print(f"Performance plots and CSV logs saved for {method} experiment.")

def train_experiment(method):
    print(f"\n--- Running Experiment with Binarization: {method.upper()} ---")
    
    # 1. Preprocessing Check
    if not os.path.exists(os.path.join(config.PROCESSED_DIR, method)):
        run_preprocessing(method)

    train_dir = os.path.join(config.PROCESSED_DIR, method, 'train')
    test_dir = os.path.join(config.PROCESSED_DIR, method, 'test')

    # 2. Data Loading (Optimized for M4 Memory)
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        validation_split=config.VALIDATION_SPLIT,
        subset="training",
        seed=42,
        image_size=config.IMAGE_SIZE,
        batch_size=config.BATCH_SIZE,
        color_mode="grayscale"
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        validation_split=config.VALIDATION_SPLIT,
        subset="validation",
        seed=42,
        image_size=config.IMAGE_SIZE,
        batch_size=config.BATCH_SIZE,
        color_mode="grayscale"
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        image_size=config.IMAGE_SIZE,
        batch_size=config.BATCH_SIZE,
        color_mode="grayscale"
    )

    # 3. Model Initialization
    model = get_model('inception_resnet_v2', num_classes=config.NUM_CLASSES)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # 4. Training (Incremental Approach - 300 Epochs)
    # Using the Dropout layers defined in model_factory.py to prevent overfitting
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.EPOCHS,
        verbose=1
    )

    # 5. Visualizing & Saving Results
    plot_history(history, method)

    print(f"\nEvaluating Model on Final Test Set ({method}):")
    results = model.evaluate(test_ds)
    print(f"Test Loss: {results[0]:.4f}, Test Accuracy: {results[1]:.4f}")
    
    # 6. Save Model for Stage 2 (Word Segmentation)
    os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)
    model.save(os.path.join(config.MODEL_SAVE_DIR, f"modi_model_{method}.h5"))

if __name__ == "__main__":
    # Execute the three experimental frameworks 
    # for m in ['otsu', 'sauvola', 'tozero']:
    for m in ['tozero']:
        train_experiment(m)