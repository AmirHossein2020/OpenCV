import cv2  # (1)=> اول باید پکیج و فرایخونیم
#=======================================================================================================
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +  
            'haarcascade_frontalface_default.xml')
# (2) => درست کنیم تا بتونیم از قابلیت تشخیص چهره استفاده کنیم face_cascade برای تشخیص چهره باید 
#=======================================================================================================
video_capture = cv2.VideoCapture(0) # (3) => در اینجا مسیر ویدیو یا تصویری که می خواهیم
# برنامه رو باهاش درست کنیم انتخاب میکنیم
#=======================================================================================================
custom_texts = [] # (4) => یک متغیره درست میکنیم تا متن های که می خواهیم داخلش ذخیره کنیم
#=======================================================================================================
while True: # (5) => برای اینکه بتونیم از ویدیو استفاده کنیم باید کد های دستوریم داخل یه حلقه بزنیم

    ret, frame = video_capture.read()
# (6) => اینجا باید با وارد کردن کد بالا ویدیو که مسیرش و انتخاب کردیم بزنیم تا ویدیو بخونه
    if not ret:
        break
# این شرط هم برای نادرست بودن مسیر که اگه غلط بود از برنامه خارج بشه
#=======================================================================================================
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) # (7) => اینجا باید اول ویدیو ای که داریم و تبدیل به 
    # سیاه و سفید کنیم
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    # و بعد چهره ای که از ویدیو میگیریم و از داخل ویدیو سیاه و سفید تشخیص بده و شناسایی کنه
#=======================================================================================================
    for (x, y, w, h) in faces: # (8) => حالا یه حلقه درست میکنیم تا هر چهره و تویی ویدیو شناسایی کرد 
        face_roi = gray[y:y+h, x:x+w]# نسبت به هر چهره متن و متفاوتی داشته باشه
#=======================================================================================================
        if len(custom_texts) < len(faces): # (9) =>حالا یه شرط درست میکنیم که هر چهره ای که شناسایی کرد
            text = "USER"
            #text = input("Enter YOUR NAME: ")# اسم اون چهره و از کاربر بگیریه
            custom_texts.append(text)# و بعد ذخیره کنه 
        elif text == text: # بعد میایم میگیم اگه اون چهره اسمش امیر حسین بود بیا روی صحفه ای
            cv2.putText(frame, f"Welcom to pragram {text}", (70, 50),
             cv2.FONT_HERSHEY_COMPLEX, 1, (210,40,0), 2)# ویدیو بهش خوشامد بگو و روی صحفه متن خوشامد گویی چاپ کن
        else:# ما اندازه فونت و رنگ و مختصاتش تویی ویدیو با استفاده از کد بالا انجام بدیم
            text = custom_texts[faces.tolist().index([x, y, w, h])]
        # در اخر شرط هم میتونیم بگیم اگه هیچکدوم از اسم ها و چهره ها نبود از تویی اسم ها و متن های که ذخیره کردیم انتخاب کن و نمایش بده
#=======================================================================================================
        font = cv2.FONT_HERSHEY_SIMPLEX # (10) => تو این مرحله میایم جایی که اسم که گرفتیم باید قرار بگیرد 
        font_scale = 1 # تعیین میکنیم و انداز ه و رنگ و مختصات که باید کنار تصویر باشه رو انجام میدیم
        font_thickess = 2
        text_size = cv2.getTextSize(text, font, font_scale, font_thickess)[0]
        text_x = x + (w - text_size[0]) // 2
        text_y = y - 10
        cv2.putText(frame, text, (text_x, text_y), font, font_scale,
                     (40,233,235), font_thickess, cv2.LINE_AA)
        # و در اخر با استفاده از کد بالا اون رویی صحفه تصویر نمایش میدیم
#=======================================================================================================
        cv2.rectangle(frame, (x,y), (x+w ,y+h), (5, 70, 20), 2)# (11) => با این دستور هر چهره ای که
        # تو تصویرر شناسایی کنه یه یه خط مربع ای دور چهره میزاره که اون شناسایی کرده
#=======================================================================================================
    cv2.imshow('Video with Detected Face', frame)# (12) => اینجا این دستور و میزنیم تا هر کاری که گفتیم
    # نشون بده و اون اجراش کنه و به نماییش بزاره وگرنه ویدیو ای یا تصویری پخش نمیشه
#=======================================================================================================
    if cv2.waitKey(1) & 0xFF == ord("q"): # (13) => با استفاده از این شرط تعیین میکنیم که هروقت
        break # روی کیبورد زدیم از برنامه خارج بشه و برنامه رو ببندهq دکه ای 
#=======================================================================================================
video_capture.release()# (14) => در اخر هم خارج از حلقه دستور های روبه رو و میزنیم تا 
cv2.destroyAllWindows()# برنامه و اجرا کنیم
#=======================================================================================================