<p align="center">
  <img width="200" alt="image" src="https://github.com/user-attachments/assets/2a632e68-d002-4c2c-9b5b-37a4185d1274">
</p>

# Leveraging Machine Learning for Precision Medicine in Ghana
Machine Learning with Graphs using RGCN for link prediction, cosine similarity and node2vec for predicting novel associations and streamlit for web interface.

## 🙌🏽 Research Question and Objective
Research indicates that machine learning for precision medicine significantly improves patient survival outcomes in developed countries. However, a noticeable gap exists in its application in developing countries, such as Ghana. Hence, there is a critical need to establish systems dedicated to delivering personalized treatments using machine learning. This research investigates the question, “can machine learning be leveraged to develop personalized plans in the context of Ghana’s personalized healthcare system?”

### 🚀 This repo contains
- Source code for drug disease association prediction
- Source code for drug disease gene/protein association prediction (with genes (**UGT1A1, PIK3CA, BRAF, NRAS, KRAS and DYPD**) from Ghanaian dataset)
- Streamlit web interface for both predictions


### 🫶🏽 Data Source
All data used in this project is from Harvard's Zitnik Lab's Precision Medicine Knowledge Graph (PrimeKG)

[LINK TO THEIR REPO](https://github.com/mims-harvard/PrimeKG)

### How to run: the notebooks
- Open the notebook in google colab (preferrably) or any IDE of choice and run the cells.

#### ❗Disclaimer
- The packages (those from pytorch geometric) were imported based on the requirements of the system I used to run the code. You wouldn't have issues in google colab, just add **kg.csv** to your drive change the path to it in the second cell. If you decided to use your gpu, then you'd have to change the imports to obtain the gpu versions of the packages.

### How to run: the interface
- Open the **web-app.py**
- In the **load_data** function, change the links to the location of the various pickle files. Example: change this **/Users/sarpongmorgan/Downloads/Capstone stuff/CapstoneProject/tests/drug-disease-association-application/drug_nodes.pkl** to **/path/to/drug_nodes.pkl** on your machine.
- create a virtual environment or download anaconda navigator and create an environment from there. [This video can help if you're stuck](https://www.youtube.com/watch?v=WLwjvWq0GWA) **watch from 33:26**
- pip install the following: streamlit, torch, pickle, pandas, io and reportlab
- Run the following command: **streamlit run "path/to/web-app.py"**

#### You should see this interface:

![image](https://github.com/user-attachments/assets/192d9b4e-ff90-4dcb-84fc-e0253b5cd6de)

![image](https://github.com/user-attachments/assets/2ad1c4d7-a09e-4606-b047-3bd082c92a87)




### Visualisations: for drug disease association prediction

<p align="center">
  <img width="508" alt="image" src="https://github.com/user-attachments/assets/32324c72-099c-4794-9758-712808b0e762">
</p>

### Visualisations: for drug disease gene association prediction
<p align="center">
  <img width="566" alt="image" src="https://github.com/user-attachments/assets/618154dc-c7f5-41e8-bda0-e4a8a6084292">
</p>


### Interesting findings from literature review:
<p align="center">
<img width="612" alt="image" src="https://github.com/user-attachments/assets/c6d8c826-9cc5-4b0c-b81e-d92b4f66f743">
</p>


