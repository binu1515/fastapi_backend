from fastapi import FastAPI, File, UploadFile
from main import pred_and_plot_image
#from plant_details import get_plant_details

app = FastAPI()

@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile):
    print("hello")
    print(file)
    pred_class = pred_and_plot_image(image_bytes=file.file.read())
    return {"filename": file.filename, "prediction": pred_class} #"data": get_plant_details(pred_class)}