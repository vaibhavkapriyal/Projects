from cryptography.fernet import Fernet
from tkinter import *
from tkinter import messagebox
from tkinter.filedialog import askopenfilename
from PIL import Image, ImageTk  # PIL for image handling
import os

# Function to generate a key for encryption
def generate_key():
    try:
        # key generation
        key = Fernet.generate_key()

        # storing the key in a file
        with open('filekey.key', 'wb') as filekey:
            filekey.write(key)
        messagebox.showinfo("", "Key generated successfully!")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred while generating the key: {str(e)}")

# Function to encrypt the file
def encrypt():
    try:
        # opening the key
        with open('filekey.key', 'rb') as filekey:
            key = filekey.read()
    
        # using the generated key
        fernet = Fernet(key)
    
        # opening the original file to encrypt
        path = entryvalue.get().strip()
        
        if not os.path.isfile(path):
            raise FileNotFoundError(f"File not found: {path}")
        
        with open(path, 'rb') as file:
            original = file.read()
        
        # encrypting the file
        encrypted = fernet.encrypt(original)
    
        # opening the file in write mode and writing the encrypted data
        with open(path, 'wb') as encrypted_file:
            encrypted_file.write(encrypted)
        
        messagebox.showinfo('', 'File encryption successful!')
    
    except FileNotFoundError as e:
        messagebox.showerror("Error", str(e))
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred while encrypting the file: {str(e)}")

# Function to decrypt the file
def decrypt():
    try:
        # opening the key
        with open('filekey.key', 'rb') as filekey:
            key = filekey.read()
        
        # using the key
        fernet = Fernet(key)
    
        # opening the encrypted file
        path = entryvalue1.get().strip()
        
        if not os.path.isfile(path):
            raise FileNotFoundError(f"File not found: {path}")
    
        with open(path, 'rb') as enc_file:
            encrypted = enc_file.read()
    
        # decrypting the file
        decrypted = fernet.decrypt(encrypted)
    
        # opening the file in write mode and writing the decrypted data
        with open(path, 'wb') as dec_file:
            dec_file.write(decrypted)
    
        messagebox.showinfo('', 'Decryption done!')
    
    except FileNotFoundError as e:
        messagebox.showerror("Error", str(e))
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred while decrypting the file: {str(e)}")

# Function to browse and select files
def browse_file(entry_widget):
    file_path = askopenfilename(title="Select a file")
    if file_path:
        entry_widget.delete(0, END)  # Clear the current text
        entry_widget.insert(0, file_path)  # Insert the selected file path

# GUI Setup
top = Tk()

top.title("File Encryption & Decryption")
top.geometry("600x500")

# Adding the background image
canvas = Canvas(top, height=500, width=600)
canvas.pack()

# Load and display the background image
bg_image = Image.open("background.jpg")  # Ensure this path is correct
bg_image = bg_image.resize((600, 500), Image.Resampling.LANCZOS)  # Resize the image to fit the window
bg_image = ImageTk.PhotoImage(bg_image)

canvas.create_image(0, 0, image=bg_image, anchor=NW)

# Set custom font and style
font_style = ("Helvetica", 12)

# Header Label
header_label = Label(top, text="File Encryption & Decryption", fg="white", bg="#34495E", font=("Helvetica", 16, "bold"))
header_label.place(x=150, y=20)

# Generate Key Button
button2 = Button(top, text="Generate Key", bg="#27AE60", fg="white", font=font_style, relief="solid", width=20, height=2, command=generate_key)
button2.place(x=200, y=80)

# Encryption Section
l = Label(top, text='Input the file path to encrypt:', fg="white", bg="#34495E", font=font_style)
l.place(x=50, y=150)

entryvalue = Entry(top, font=font_style, width=40)
entryvalue.place(x=50, y=180)

browse_button_encrypt = Button(top, text="Browse", bg="#2980B9", fg="white", font=font_style, width=10, command=lambda: browse_file(entryvalue))
browse_button_encrypt.place(x=450, y=175)

button_encrypt = Button(top, text="Encrypt", bg="#27AE60", fg="white", font=font_style, relief="solid", width=20, height=2, command=encrypt)
button_encrypt.place(x=200, y=220)

# Decryption Section
l1 = Label(top, text='Input the file path to decrypt:', fg="white", bg="#34495E", font=font_style)
l1.place(x=50, y=280)

entryvalue1 = Entry(top, font=font_style, width=40)
entryvalue1.place(x=50, y=310)

browse_button_decrypt = Button(top, text="Browse", bg="#2980B9", fg="white", font=font_style, width=10, command=lambda: browse_file(entryvalue1))
browse_button_decrypt.place(x=450, y=305)

button_decrypt = Button(top, text="Decrypt", bg="#27AE60", fg="white", font=font_style, relief="solid", width=20, height=2, command=decrypt)
button_decrypt.place(x=200, y=350)

# Exit Button
exit_button = Button(top, text="Exit", bg="#E74C3C", fg="white", font=font_style, relief="solid", width=20, height=2, command=top.destroy)
exit_button.place(x=200, y=420)

top.mainloop()
