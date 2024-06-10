from cryptography.fernet import Fernet
from tkinter import *
from tkinter import messagebox

def generate_key():
    # key generation
    key = Fernet.generate_key()

    # string the key in a file
    with open('filekey.key', 'wb') as filekey:
        filekey.write(key)
    messagebox.showinfo( "", "Key generated successfully!")


def encrypt():
     # opening the key
    with open('filekey.key', 'rb') as filekey:
        key = filekey.read()
  
    # using the generated key
    fernet = Fernet(key)
  
    # opening the original file to encrypt
    path = entryvalue.get()

    
    with open(path, 'rb') as file:
        original = file.read()
      
    # encrypting the file
    encrypted = fernet.encrypt(original)
  
    # opening the file in write mode and 
    # writing the encrypted data
    with open(path, 'wb') as encrypted_file:
        encrypted_file.write(encrypted)
        
    messagebox.showinfo('','File encryption Sucessfull.....\n')


def decrypt():
    # opening the key
    with open('filekey.key', 'rb') as filekey:
        key = filekey.read()
        
    # using the key
    fernet = Fernet(key)
  
    # opening the encrypted file
    path = entryvalue1.get()

    with open(path, 'rb') as enc_file:
        encrypted = enc_file.read()
  
    # decrypting the file
    decrypted = fernet.decrypt(encrypted)
  
    # opening the file in write mode and
    # writing the decrypted data
    with open(path, 'wb') as dec_file:
        dec_file.write(decrypted)

    messagebox.showinfo('','Decryption done.......\n')


top=Tk()

top.title("File Encryption")

top.geometry("500x500")

top.configure(bg='black')

# Generate key
button2 = Button(top,text="Generate Key",bg="green", fg="yellow",command=generate_key) 
button2.pack(padx=5, pady=15)

#Encryption
l = Label(top,text='Input the path',bg="red", fg="yellow")
l.pack(padx=5, pady=20) 

entryvalue = Entry(top)
entryvalue.pack()


button = Button(top,text="Encrypt",bg="green", fg="yellow",command=encrypt) 
button.pack(padx=5, pady=20)

#Decryption
l1 = Label(top,text='Input the path',bg="red", fg="yellow")
l1.pack(padx=5, pady=20) 

entryvalue1 = Entry(top) 
entryvalue1.pack()


button1 = Button(top,text="Decrypt",bg="green", fg="yellow",command=decrypt) 
button1.pack(padx=5, pady=30)

exit_button = Button(top, text="Exit",bg="blue", fg="white", command=top.destroy)
exit_button.pack(padx=5, pady=35)

top.mainloop()

