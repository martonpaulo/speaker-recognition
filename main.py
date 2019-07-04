from tkinter import *

  
class Application(Frame):

	def __init__(self, parent=None):
		Frame.__init__(self,parent)
		self.parent = parent
		self.make_widgets()


	def make_widgets(self):

		self.winfo_toplevel().title("Aluizium 2.1")


		description = []
		description.append("Speaker recognition software Aluizium 2.1")
		description.append("Made by: Carlos Eduardo, Marton Paulo, Matheus Barbosa and Victor Godoy")
		description = '\n'.join(description)


		self.msg = Label(self.parent, text=description)
		#self.msg["font"] = ("Verdana", "10")
		#self.msg.grid(row=0, column=0, padx= 10, pady=10)
		self.msg.grid(row=0, column=0, columnspan=3, padx=25, pady=25)



		self.exit = Button(self.parent, text="New speaker")
		self.exit["command"] = self.parent.quit
		self.exit.grid(row=1, column=0, padx=25, pady=25)

		self.exit = Button(self.parent, text="Recognize speaker")
		self.exit["command"] = self.parent.quit
		self.exit.grid(row=1, column=1, padx=25, pady=25)

		self.exit = Button(self.parent, text="Exit")
		self.exit["command"] = self.parent.quit
		self.exit.grid(row=1, column=2, padx=25, pady=25)


		



root = Tk()
root.resizable(width=False, height=False)
app = Application(root)
root.mainloop()
