# Liquid-Neural-Networks-LNNs-Classification

- Liquid Neural Networks (LNNs) Classification, Clustering, and Regression

<div align="justify">

- This code implements a neural network using PyTorch, specifically designed for handling the Iris dataset through various machine learning tasks. It introduces a custom recurrent neural network architecture called LTCNetwork, built around a specialized cell named LTCCell, which features dynamic time constants (𝜏 τ) to adjust the neuron's response behavior. This architecture is utilized in three main tasks: first, it performs the classification of Iris species, calculates the average accuracy across multiple trials, and generates detailed performance metrics such as confusion matrices and classification reports. Second, it leverages the model's hidden states to conduct clustering using KMeans, visualizing the clustering outcomes and evaluating their quality with a silhouette score. Lastly, the network is adapted to perform regression to predict sepal length, using other features of the dataset, and evaluates the regression performance by calculating and reporting the mean squared error and correlation coefficient. The script demonstrates the versatility and capability of the LTCNetwork in adapting to different types of data-driven tasks while managing temporal dynamics effectively.

</div>

![image](https://github.com/user-attachments/assets/4a689ad5-a941-43f8-bc6c-d5f1ce175c4d)

![image](https://github.com/user-attachments/assets/fc386c99-7b11-4f8e-b699-7535cc7dfe8b)
