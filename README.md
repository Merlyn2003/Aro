Install all the required moduled and packages from the requirements.txt through the command
                            pip install requirements.txt
in the vscode terminal.

Download the Quantized BioMistral LLM "BioMistral-7B.Q4_K_M.gguf" from the https://huggingface.co/MaziyarPanahi/BioMistral-7B-GGUF/tree/main in your working directory.

To upload all the pdfs in the data folder, you need to run the vector_db.py. Since we use qdrant, you will need to run the docker for the port 6333 before executing the vector_db.py.
After setting up the docker, execute the vector_db: python vector_db.py

It will take some time to upload all the files from the data folder.

After successfully executing vector_db.py,now execute the Appln.py.

The Appln.py will run on port 5000.

