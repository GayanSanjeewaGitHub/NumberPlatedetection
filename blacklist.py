import os
import cv2
import datetime

import smtplib, ssl
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email.mime.image import MIMEImage
from email import encoders

port = 587  # For SSL 465 else 587

# change to your email account settings
smtp_server = 'smtp.gmail.com' # mail smtp
sender = 'xxxxxxx@gmail.com'  # Enter your address
receiver = 'xxxxxx@gmail.com'  # Enter receiver address
password = 'ixxxxxxuojzq' # mail password

# this password is not your gmail account password, this password has to create via gmail account setting , under 'App Password'(search for it)
#need to add two steps authenticator ,and other chnages in sending email, refer some docs to setup this seperately from your account
# this is for gmail by the way

blacklist = [] # blacklist list
infile = open('blacklist.txt','r') # open blacklist.txt

for line in infile: # read blacklist.txt
    line = line.replace("\n", "")
    blacklist.append(line)

infile.close() # close blacklist.txt

def search(img,lproi,cam,now,text): # blacklist search
    timestamp=now.strftime('%d.%m.%Y. %H:%M:%S')
    for i in range(len(blacklist)):
        if blacklist[i] == text:
            print(timestamp+' Possible ANPR match: ' + text + ' at location '+cam)

            msg = MIMEMultipart('related')
            msg['Subject'] = 'Possible ANPR match at location ' +cam
            msg['From'] = sender
            msg['To'] = receiver
            msg.preamble = 'Multi-part message in MIME format.'

            msgAlternative = MIMEMultipart('alternative')
            msg.attach(msgAlternative)

            msgText = MIMEText(timestamp+' Possible ANPR match: '+text+' at location '+cam)
            msgAlternative.attach(msgText)

            msgText = MIMEText(timestamp+'<br><br>Possible ANPR match at location '+cam+'<h1>'+text+'</h1><img src="cid:lproi"><br><br><img src="cid:img">', 'html')
            msgAlternative.attach(msgText)

            cv2.imwrite('mail/lproi.jpg',lproi)
            cv2.imwrite('mail/img.jpg',img)
            
            path="./mail/"

            try:
                filelist = ["lproi.jpg", "img.jpg"]
                for file in filelist:
                    filename=(os.path.splitext(file)[0])

                    attachment = open(path + file, "rb") #   open the file in "read binary" mode
                    part = MIMEBase('application', 'octet-stream')
                    part.set_payload((attachment).read())
                    part.add_header('Content-ID', '<'+filename+'>')
                    encoders.encode_base64(part) # Encode file in ASCII characters to send by email
                    part.add_header('Content-Disposition', "attachment; filename= %s" % file) # Add header as key/value pair to attachment part
                    msg.attach(part)
            except:
                print ("Blacklist error: Failed to add attachment!")

            context = ssl.create_default_context()
            try:
                server=smtplib.SMTP(smtp_server, port)
                server.ehlo()  # Can be omitted
                server.starttls(context=context)
                server.ehlo()  # Can be omitted
                server.login(sender, password)
                server.sendmail(sender, receiver, msg.as_string())
                server.close()
                print ('Blacklist: Email sent successfully!')
            except Exception as ex:
                print ('Blacklist error while sending the email message: ',ex)
