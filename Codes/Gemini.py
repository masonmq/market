from google import genai
from google.genai import types
GEMINI_API_KEY = ""

# read the image input post 
with open('/Users/samiramalek/Documents/SMPD/Agentic_Market_Model/SMPD/train/3@N61/373535.jpg', 'rb') as f:
    image_bytes = f.read()

# read the examples
examples_path = "/Users/samiramalek/Documents/SMPD/Agentic_Market_Model/SMPD/Few_shot/FewShot_Examples_Social&People.md"
with open(examples_path, "r", encoding="utf-8") as f:
    examples = f.read()

Agent_Instruction_Prompt = f"""
### [DATA INPUT]: 
=== Image path (row index) ===

  "image_path": "train/27882@N8/194355.jpg",
  "row_index": 32555


=== Category ===

  "Category": "Urban",
  "Concept": "wallporn",
  "Pid": "194355",
  "Uid": "27882@N8",
  "Subcategory": "StreetArt"


=== Text ===

  "Alltags": "streetart graffiti urbanart grafite artederua arteurbana fromthestreets ufoo tvstreetart wallporn sampagraffiti ingf misturaurbana instagrafite instagraffiti uploaded:by=flickstagram rsagraffiti dsbgraff muralsdaily brarts instagram:photo=937616898411796705321503909",
  "Pid": "194355",
  "Uid": "27882@N8",
  "Mediatype": "photo",
  "Title": "Arte de @ufo__o #graffiti #grafite #streetart..."


=== Temporal-spatial ===

  "Postdate": "1425963866",
  "Uid": "27882@N8",
  "Pid": "194355",
  "Longitude": "",
  "Geoaccuracy": "0",
  "Latitude": "0",
  "Postdate_datetime": "2015-03-10 01:04:26"


=== User profile ===

  "Uid": "27882@N8",
  "Pid": 194355,
  "photo_firstdate": null,
  "photo_count": 1813.0,
  "ispro": 0.0,
  "timezone_offset": null,
  "photo_firstdatetaken": "2013-04-02 17:52:16",
  "timezone_id": null,
  "user_description": null,
  "location_description": null


=== Additional information ===

  "Mediastatus": "ready",
  "Pathalias": "None",
  "Ispublic": "1",
  "Pid": "194355",
  "Uid": "27882@N8"

Here are some examples of how to evaluate a post:
{examples}

You are a senior social media analyst.You evaluate the post based on the provided image and metadata.

The metadata includes:

### Category

* `Category`: the primary category of the post (11 classes).
* `Subcategory`: one of 77 second-level categories.
* `Concept`: one of 668 different descriptions.

### Text

* `Title`: the title of the post defined by the user.
* `Mediatype`: the type of attached media, including `photo` and `video`.
* `Alltags`: user-defined tags.

### Temporal-spatial

* `Postdate`: the publication timestamp of the post. It can be converted to datetime using the following Python code:

```python
import time
timestamp = 1457068974
timeArray = time.localtime(timestamp)
datetime = time.strftime("%Y-%m-%d %H:%M:%S", timeArray)
```

* `Latitude`: latitude in the range [-90, 90]. Values beyond 6 decimal places are truncated.
* `Longitude`: longitude in the range [-180, 180]. Values beyond 6 decimal places are truncated.
* `Geoaccuracy`: accuracy level of the location information.

  * World level ≈ 1
  * Country ≈ 3
  * Region ≈ 6
  * City ≈ 11
  * Street ≈ 16
    The valid range is 1–16, with a default value of 16 if unspecified.

### User Profile

* `Photo_firstdate`: the upload date of the user's first photo.
* `Photo_count`: the total number of photos posted by the user.
* `Ispro`: whether the user is a Pro member.
* `Photo_firstdatetaken`: the capture date of the user's first photo.
* `Timezone_offset`: the user's timezone offset.
* `User description`: descriptive information about the user.
* `Location description`: descriptive information about the user's location.

### Additional Information

* `Pathalias`: the user-defined path alias.
* `Ispublic`: indicates whether the post has public read permissions.
* `Mediastatus`: indicates whether the media is ready to be accessed by others.

Your task is to predict the **popularity_level** of a post relative to typical user-generated photo content on a large social media platform, using the provided image and metadata.

Do NOT claim or assume real view counts or external engagement data.

### Scoring Rubric (1–5)

* `1` = Very low expected engagement
* `2` = Below average engagement
* `3` = Average content
* `4` = Above average engagement
* `5` = High viral potential

###the current market estimation is “2”. 

###Your previous estimation for popularity level of this post was “3”.

### Output format

Return ONLY valid integer from 1 to 5 (no markdown, no explanation):


  "popularity_level": <integer from 1 to 5>
"""

client = genai.Client(api_key=GEMINI_API_KEY)
response = client.models.generate_content(
    #model='gemini-3.5-flash',
    model='gemini-2.5-flash',
    contents=[
      types.Part.from_bytes(
        data=image_bytes,
        mime_type='image/jpeg',
      ),
      Agent_Instruction_Prompt
    ]
  )

print(response.text)

