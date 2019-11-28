from google_images_download import google_images_download

response = google_images_download.googleimagesdownload()

argumen = [
        {
            "keywords_from_file": "top_1k_actors.txt",
            "limit": 20,
            "size": "medium"
        }
    ]

for arg in argumen:
    absolute_image_paths = response.download(arg)