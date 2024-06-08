from tkinter import *  
from tkinter import Menu 
from tkinter import filedialog
import numpy as np
from tkinter import ttk
from all_tests import *
from all_tests import *
import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
from modeling_statitic import modeling_criterion
from do_p import find_power_of
import pandas as pd
import time
    
krit_name = ["Крит.показ-ти Андерсона-Дарлинга",
         "Крит.показ-ти Аткинсона ПолуНорм(-0.25)",
         "Крит.показ-ти Аткинсона ПолуНорм(-0.5)",
         "Крит.показ-ти Аткинсона ПолуНорм(-0.75)",
         "Крит.показ-ти Аткинсона ПолуНорм(-0.99)",
         "Крит.показ-ти Аткинсона ПолуНорм(0)",
         "Крит.показ-ти Аткинсона ПолуНорм(0.25)",
         "Крит.показ-ти Аткинсона ПолуНорм(0.5)",
         "Крит.показ-ти Аткинсона ПолуНорм(0.75)",
         "Крит.показ-ти Аткинсона ПолуНорм(0.99)",
         "Крит.показ-ти Садепура r = 2",
         "Крит.показ-ти Барингхауса-Хензе(0.1)",
         "Крит.показ-ти Барингхауса-Хензе(0.5)",
         "Крит.показ-ти Барингхауса-Хензе(1)",
         "Крит.показ-ти Барингхауса-Хензе(1.5)",
         "Крит.показ-ти Барингхауса-Хензе(10)",
         "Крит.показ-ти Барингхауса-Хензе(2.5)",
         "Крит.показ-ти Барингхауса-Хензе(5)",
         "Корреляционный крит.показ-ти",
         "Корреляционный крит.показ-ти аппроксимация",
         "Крит.показ-ти Кокса-Оукса",
         "Крит.показ-ти Крамера-Мизеса",
         "Крит.показ-ти Крамера-Мизеса-Смирнова MRL",
         #"Крит.показ-ти Заманзаде",
         "Крит.показ-ти Дешпанде(0.1)",
         "Крит.показ-ти Дешпанде(0.2)",
         "Крит.показ-ти Дешпанде(0.3)",
         "Крит.показ-ти Дешпанде(0.4)",
         "Крит.показ-ти Дешпанде(0.44)",
         "Крит.показ-ти Дешпанде(0.5)",
         "Крит.показ-ти Дешпанде(0.6)",
         "Крит.показ-ти Дешпанде(0.7)",
         "Крит.показ-ти Дешпанде(0.8)",
         "Крит.показ-ти Дешпанде(0.9)",
         #"Крит.показ-ти Ибрагими",
         "Крит.показ-ти Эппса-Палли",
         "Крит.показ-ти Эпштейна",
         "Крит.показ-ти Фишера",
         "Крит.показ-ти Фортиана и Гране",
         "Крит.показ-ти Фроцини",
         "Крит.показ-ти Джини",
         "Крит.показ-ти Гнеденко(0.1)",
         "Крит.показ-ти Гнеденко(0.2)",
         "Крит.показ-ти Гнеденко(0.3)",
         "Крит.показ-ти Гнеденко(0.4)",
         "Крит.показ-ти Гнеденко(0.5)",
         "Крит.показ-ти Гнеденко(0.6)",
         "Крит.показ-ти Гнеденко(0.7)",
         "Крит.показ-ти Гнеденко(0.8)",
         "Крит.показ-ти Гнеденко(0.9)",
         "Крит.показ-ти Гринвуда",
         "Крит.показ-ти Харриса(0.1)",
         "Крит.показ-ти Харриса(0.2)",
         "Крит.показ-ти Харриса(0.25)",
         "Крит.показ-ти Харриса(0.3)",
         "Крит.показ-ти Харриса(0.4)",
         "Крит.показ-ти Хегази-Грина T1",
         "Крит.показ-ти Хегази-Грина T2",
         "Крит.показ-ти Хензе(0.025)",
         "Крит.показ-ти Хензе(0.1)",
         "Крит.показ-ти Хензе(0.5)",
         "Крит.показ-ти Хензе(1)",
         "Крит.показ-ти Хензе(1.5)",
         "Крит.показ-ти Хензе(2.5)",
         "Крит.показ-ти Хензе(5)",
         "Крит.показ-ти Хензе-Мейнтаниса L (0.1)",
         "Крит.показ-ти Хензе-Мейнтаниса L (0.5)",
         "Крит.показ-ти Хензе-Мейнтаниса L (0.75)",
         "Крит.показ-ти Хензе-Мейнтаниса L (1)",
         "Крит.показ-ти Хензе-Мейнтаниса L (1.5)",
         "Крит.показ-ти Хензе-Мейнтаниса L (2.5)",
         "Крит.показ-ти Хензе-Мейнтаниса L (5)",
         #"Крит.показ-ти Хензе-Мейнтаниса T1 (1.5)",
         #"Крит.показ-ти Хензе-Мейнтаниса T1 (2.5)",
         #"Крит.показ-ти Хензе-Мейнтаниса T2 (1.5)",
         #"Крит.показ-ти Хензе-Мейнтаниса T2 (2.5)",
         "Крит.показ-ти Хензе-Мейнтаниса W1 (0.5)",
         "Крит.показ-ти Хензе-Мейнтаниса W1 (0.75)",
         "Крит.показ-ти Хензе-Мейнтаниса W1 (1)",
         "Крит.показ-ти Хензе-Мейнтаниса W1 (1.5)",
         "Крит.показ-ти Хензе-Мейнтаниса W1 (2.5)",
         "Крит.показ-ти Хензе-Мейнтаниса W2 (0.5)",
         "Крит.показ-ти Хензе-Мейнтаниса W2 (0.75)",
         "Крит.показ-ти Хензе-Мейнтаниса W2 (1)",
         "Крит.показ-ти Хензе-Мейнтаниса W2 (1.5)",
         "Крит.показ-ти Хензе-Мейнтаниса W2 (2.5)",
         "Крит.показ-ти Холландера-Прошана",
         "Крит.показ-ти L2",
         "Крит.показ-ти Джексона",
         "Крит.показ-ти Климко-Антла",
         #"Крит.показ-ти кернеел",
         "Крит.показ-ти Кимбера-Мичела",
         "Крит.показ-ти Клара(1)",
         "Крит.показ-ти Клара(10)",
         "Крит.показ-ти Кочара",
         #"Крит.показ-ти Колмогоровоа мрл",
         "Крит.показ-ти Колмогорова-Смирнова",
         "Крит.показ-ти Купера",
         #Крит.показ-ти Лоулесса",
         "Крит.показ-ти Мадукайфе",
         "Крит.показ-ти наибольшего интервала",
         "Крит.показ-ти Монтазери и Тораби",
         "Крит.показ-ти Морана(норм)",
         "Крит.показ-ти Лоуренса(0.1)",
         "Крит.показ-ти Лоуренса(0.25)",
         "Крит.показ-ти Лоуренса(0.5)",
         "Крит.показ-ти Лоуренса(0.75)",
         "Крит.показ-ти Лоуренса(0.9)",
         "Крит.показ-ти Шапиро-Уилка",
         "Крит.показ-ти Шапиро-Уилка We0",
         "Крит.показ-ти Шермана/Пиэтра",
         #"Крит.показ-ти Sn(осн.на Gini)",
         #"Крит.показ-ти Тико",
         #"Крит.показ-ти Тораби1",
         #"Крит.показ-ти Тораби2",
         #"Крит.показ-ти U1",
         #"Крит.показ-ти U2",
         #"Крит.показ-ти N2",
         #"Крит.показ-ти Ватсона",
         "Крит.показ-ти Вонга-Вонга",
         #"Крит.показ-ти Жанга Za",
         "Крит.показ-ти Ахсануллаха",
         "Крит.показ-ти Россберга"
         
]

krit_exp = [0] * len(krit_name)
file_names = []
column_values = [0] * len(krit_name)
selected_criteria =[]

def generate_data(distribution_name, sample_size, seed, filename):
    #np.random.seed(seed)
    if distribution_name == "Экспоненциальное с масштабом 1.0000 со сдвигом 0.0000":
        sample = np.random.exponential(scale=1.0, size=sample_size)
    elif distribution_name == "Логарифмически(ln) Нормальное с масштабом 1.0000 со сдвигом 0.0000 с масштабом 1.0000 со сдвигом 0.0000":
        sample = np.random.lognormal(mean=0.0, sigma=1.0, size=sample_size)
    elif distribution_name == "Вейбулла (0.8000) с масштабом 1.0000 со сдвигом 0.0000":
        shape = 0.8  # shape parameter (a)
        scale = 1.0  # scale parameter (this is just a multiplier in numpy's weibull)
        shift = 0.0  # shift parameter (if necessary)
        sample = np.random.weibull(a=shape, size=sample_size) * scale + shift

    elif distribution_name == "Вейбулла (1.2000) с масштабом 1.0000 со сдвигом 0.0000":
        sample = np.random.weibull(a=1.2, size=sample_size)
    else:
        print("Неизвестное распределение")
        return

    with open(filename, 'w', encoding='utf-8') as f:
        f.write(distribution_name + '\n')
        f.write(str(sample_size) + '\n')
        for s in sample:
            f.write(str(s) + '\n')

def create_modeling_window(window1):
    window = Toplevel(window1)
    window.title("Окно моделирования")
    window.geometry('560x260')

    def update_values(*args):
        ras_names = ("Экспоненциальное с масштабом 1.0000 со сдвигом 0.0000", 
                    "Логарифмически(ln) Нормальное с масштабом 1.0000 со сдвигом 0.0000 с масштабом 1.0000 со сдвигом 0.0000",
                    "Вейбулла (0.8000) с масштабом 1.0000 со сдвигом 0.0000", 
                    "Вейбулла (1.2000) с масштабом 1.0000 со сдвигом 0.0000")
        zag_names = ("H0", "H1", "H2", "H3")
        file_names = ("Эксп(0.0000,1.0000).dat", 
                    "Логарифмически(ln) N(0.0000,1.0000).dat", 
                    "Вей (0.8000,1.0000,0.0000).dat", 
                    "Вей (1.2000,1.0000,0.0000).dat")
        val = filename.get()
        index = ras_names.index(val)
        
        file_zag.set(zag_names[index])
        file_label.set(file_names[index])
    frame = LabelFrame(window, text="Задать параметры выборки")
    frame.pack(fill="both", expand=True, padx=10, pady=10)
    label_0 = Label(frame, text="Количество наблюдений")
    label_0.grid(row=0, column=0, padx=10, pady=5, sticky=W)

    sample_size = Spinbox(frame, from_=0, to=1000, increment=1, width=10, textvariable= IntVar(value=100))
    sample_size.grid(row=0, column=1, padx=10, pady=5, sticky=W)

    label_1 = Label(frame, text="Начальное значение ГСЧ")
    label_1.grid(row=0, column=1, padx=110, pady=5, sticky=W)

    seed = Spinbox(frame, from_=0, to=1000, increment=1, width=10, textvariable= IntVar(value=100))
    seed.grid(row=0, column=1, padx=270, pady=5, sticky=W)


    label_3 = Label(frame, text="Заголовок ")
    label_3.grid(row=1, column=0, padx=10, pady=5, sticky=W)

    file_zag = StringVar()
    entry_filename_0 = Entry(frame, width=55, textvariable=file_zag)
    entry_filename_0.grid(row=1, column=1, padx=10, pady=5, sticky=W)

    label_4 = Label(frame, text="Выбрать выборку")
    label_4.grid(row=2, column=0, padx=10, pady=5, sticky=W)

    filename = StringVar()
    combobox = ttk.Combobox(frame, textvariable=filename,state='readonly', width=52)
    combobox.grid(row=2, column=1, padx=10, pady=5, sticky=W)

    label_5 = Label(frame, text= "Название файла")
    label_5.grid(row=3, column=0, padx=10, pady=5, sticky=W)
    file_label = StringVar()
    entry_filename = Entry(frame, width=55, textvariable=file_label)
    entry_filename.grid(row=3, column=1, padx=10, pady=5, sticky=W)
    distribution_values = ("Экспоненциальное с масштабом 1.0000 со сдвигом 0.0000", 
                          "Логарифмически(ln) Нормальное с масштабом 1.0000 со сдвигом 0.0000 с масштабом 1.0000 со сдвигом 0.0000",
                          "Вейбулла (0.8000) с масштабом 1.0000 со сдвигом 0.0000", 
                          "Вейбулла (1.2000) с масштабом 1.0000 со сдвигом 0.0000")
    file_zag = StringVar()
    file_zag.set("H0")
    entry_filename_0 = Entry(frame, width=55, textvariable=file_zag)
    entry_filename_0.grid(row=1, column=1, padx=10, pady=5, sticky=W)

    filename = StringVar()
    distribution_name = ttk.Combobox(frame, textvariable=filename,state='readonly', width=52)
    distribution_name['values'] = distribution_values
    distribution_name.current(0)
    distribution_name.grid(row=2, column=1, padx=10, pady=5, sticky=W)
    filename.trace("w", update_values)

    file_label = StringVar()
    file_label.set("Эксп(0.0000,1.0000).dat")
    entry_filename = Entry(frame, width=55, textvariable=file_label)
    entry_filename.grid(row=3, column=1, padx=10, pady=5, sticky=W)

    modeling_button = Button(frame, text="Моделирование", command=lambda:generate_data(distribution_name.get(), int(sample_size.get()), int(seed.get()), entry_filename.get()))
    modeling_button.grid(row=4, column=1, padx=10, pady=5, sticky=W)
    
class FileSelector(tk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.selected_files = []

        # Create widgets
        self.select_button = tk.Button(self, text="Выбрать файл", command=self.select_file)
        self.combobox = ttk.Combobox(self, width=80, state="readonly", values=file_names)

        # Place widgets
        self.select_button.pack(side="left", padx=(0, 5))
        self.combobox.pack(side="left", fill="x", expand=True)

    def select_file(self):
        filename = filedialog.askopenfilename()
        if filename:
            self.selected_files.append(filename)
            self.combobox["values"] = self.selected_files
            self.combobox.current(len(self.selected_files) - 1)
            file_names.append(filename)  # Добавляем имя файла в глобальный список

def create_exp_window(window):
    global krit_exp
    global krit_name
    global file_names

    def on_frame_configure(event):
        canvas.configure(scrollregion=canvas.bbox("all"))

    def on_canvas_configure(event):
        canvas.itemconfig(scrollable_frame_id, width=event.width)
        canvas.configure(scrollregion=canvas.bbox("all"))

    def on_mousewheel(event):
        canvas.yview_scroll(-1*(event.delta//120), "units")

    window1 = tk.Toplevel(window)
    
    window1.title("Проверка на показательность")
    window1.geometry('800x800')
    window.withdraw()
    
    def select_all():
        if var1.get() == 1:
            for var in var_list:
                var.set(1)
        else:
            for var in var_list:
                var.set(0)
    
    def change_krit_exp(var_list):
        for i in range(len(var_list)):
            krit_exp[i] = var_list[i].get()

    def close_window(var_list):
        change_krit_exp(var_list)
        window1.destroy()
        window.deiconify()
    
    def check_all_selected(*args):
        if all(var.get() == 1 for var in var_list):
            var1.set(1)
        else:
            var1.set(0)
                
    
    def do_test(window, file_name, var_list):
        change_krit_exp(var_list)
        print(file_name)
        create_exp_result(window, file_name)
        
    var1 = tk.IntVar()
    
    file_selector = FileSelector(window1)
    file_selector.pack(anchor="w", padx=(10, 0), pady=10)

    select_all_checkbutton = tk.Checkbutton(window1, text="Выбрать все", variable=var1, command=select_all)
    select_all_checkbutton.pack(anchor="w", padx=(10, 0), pady=10)

    var_list = []

    # Создание интерфейса списка элементов
    outer_frame = ttk.Frame(window1, borderwidth=2, relief="groove")
    outer_frame.pack(fill="both", expand=True, padx=10, pady=10)

    canvas = tk.Canvas(outer_frame)
    canvas.pack(side="left", fill="both", expand=True)
    
    scrollbar = ttk.Scrollbar(outer_frame, orient="vertical", command=canvas.yview)
    scrollbar.pack(side="right", fill="y")
    
    canvas.configure(yscrollcommand=scrollbar.set)

    scrollable_frame = ttk.Frame(canvas)
    scrollable_frame_id = canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")

    var_list = [IntVar() for _ in range(len(krit_name))]
    for i in range(len(var_list)):
        var_list[i].set(krit_exp[i])
        var_list[i].trace('w', check_all_selected)
        
    for element in krit_name:
        index = krit_name.index(element)
        frame = ttk.Frame(scrollable_frame)
        frame.pack(fill="x")
        label = tk.Label(frame, text= element)
        label.pack(side="left")
        checkbox = tk.Checkbutton(frame, variable=var_list[index])
        checkbox.pack(side="right")

    canvas.bind_all("<MouseWheel>", on_mousewheel)

    style = ttk.Style()
    style.configure("Custom.TFrame", borderwidth=0)
    outer_frame.configure(style="Custom.TFrame")

    scrollable_frame.bind("<Configure>", on_frame_configure)
    canvas.bind("<Configure>", on_canvas_configure)

    check_button = tk.Button(window1, text="Проверить", command=lambda: do_test(window1, file_selector.combobox.get(), var_list))
    check_button.pack(pady=10)
    
    window1.protocol('WM_DELETE_WINDOW', lambda: close_window(var_list))

def create_exp_result(window1, text):
    def close1_window():
        window2.destroy()
        window1.deiconify()
    global krit_exp
    global krit_name
    print(text)
    #text = "C:/Users/marse/Desktop/dimp/Эксп(0.0000,1.0000)_1234567.dat"
    # проверяем выбран ли файлы
    name_s = []
    result = []
    stats = []
    p_value = []
    if(text != ""):
        with open(text, 'r') as f:
            lines = f.readlines()
        lines = lines[2:]
        data = np.array([float(line.strip()) for line in lines])
        for j in range(len(krit_exp)):
            if(krit_exp[j] == 1):               
                a, b, c = test(krit_name[j], data)
                name_s.append(krit_name[j])
                result.append(c)
                stats.append(a)
                p_value.append(b)
                print(a,b,c)   
        window2 = Toplevel(window1)
        window2.title("Результаты критериев")
        window2.geometry('1070x770')
        window1.withdraw()
        
        frame = ttk.Frame(window2)
        frame.pack(fill="both", expand=True)
        # Создание таблицы
        tree = ttk.Treeview(frame)
        tree["columns"] = ("Критерий", "Гипотеза", "т-статистика", "p-значения")
        tree.column("#0", width=0, stretch=tk.NO)
        tree.column("Критерий", anchor=tk.CENTER, stretch=tk.YES)  
        tree.heading("Гипотеза", anchor=tk.CENTER)
        tree.column("т-статистика", anchor=tk.CENTER, stretch=tk.YES)  
        tree.column("p-значения", anchor=tk.CENTER, stretch=tk.YES)  
        tree.heading("#0", text="", anchor=tk.CENTER)
        tree.heading("Критерий", text="Критерий", anchor=tk.CENTER)
        tree.heading("Гипотеза", text="Гипотеза", anchor=tk.CENTER)
        tree.heading("т-статистика", text="т-статистика", anchor=tk.CENTER)
        tree.heading("p-значения", text="p_value", anchor=tk.CENTER)

        for i in range(len(name_s)):
            tree.insert(parent='', index='end', iid=i, text="", values=(name_s[i], result[i], stats[i], p_value[i]))

        tree.pack(fill='both', expand=True)
        window2.protocol('WM_DELETE_WINDOW', lambda: close1_window())

def create_statistic_window(_window):
    global selected_criteria
    global column_values
    def on_main_window_close():
        global column_values
        # Сохранение значений второго столбца перед закрытием окна
        column_values = [criteria_table.item(item, "values")[1] for item in criteria_table.get_children()]
        root.destroy()


    def on_table_click(event):
        global selected_criteria
        global column_values

        item = criteria_table.selection()[0]
        criteria_table.focus(item)
        selected_criteria_name = criteria_table.item(item, "values")[0]
        current_value = criteria_table.item(item, "values")[1]

        if current_value == 0:  # Если не выбран
            new_value = 1  # Установить флажок
            if selected_criteria_name not in selected_criteria:
                selected_criteria.append(selected_criteria_name)
        else:
            new_value = 0  # Снять флажок
            if selected_criteria_name in selected_criteria:
                selected_criteria.remove(selected_criteria_name)

        criteria_table.set(item, "Value", new_value)

        # Обновление значений второго столбца в глобальной переменной
        column_values = [criteria_table.item(item, "values")[1] for item in criteria_table.get_children()]

        # Проверяем, если все критерии выбраны, снимаем флажок с "Выбрать все"
        check_select_all()


    def select_all():
        all_selected = all(criteria_table.item(item, "values")[1] == 1 for item in criteria_table.get_children())
        if all_selected:
            for item in criteria_table.get_children():
                criteria_name = criteria_table.item(item, "values")[0]
                selected_criteria.remove(criteria_name)
                criteria_table.set(item, "Value", 0)  # Убираем флажок
        else:
            selected_criteria.clear()  # Очищаем список выбранных критериев
            for item in criteria_table.get_children():
                criteria_name = criteria_table.item(item, "values")[0]
                selected_criteria.append(criteria_name)
                criteria_table.set(item, "Value", 1)  # Устанавливаем флажок

        check_select_all()

    def check_select_all():
        all_selected = all(criteria_table.item(item, "values")[1] == 1 for item in criteria_table.get_children())
        select_all_var.set(1) if all_selected else select_all_var.set(0)

    root = Toplevel(_window)
    root.title("Моделирование статистик")
    root.geometry('550x400')

    # Флажок для выбора всех критериев
    select_all_var = tk.IntVar()
    select_all_checkbox = ttk.Checkbutton(root, text='Выбрать все', variable=select_all_var, command=select_all)
    select_all_checkbox.grid(row=0, column=0, sticky=tk.W)
    
    flag = 1
    for i in column_values:
        if(i == 0):
            flag = 0
    select_all_var.set(flag)

    criteria_frame = tk.Frame(root)
    criteria_frame.grid(row=1, column=0, columnspan=3)

    global criteria_table
    criteria_table = ttk.Treeview(criteria_frame, columns=('Критерий', 'Value'), show="headings")
    criteria_table.heading('Критерий', text='Критерий')
    criteria_table.heading('Value', text='Нажмите чтобы выбрать')
    criteria_table.column('Критерий', width=300)
    criteria_table.column('Value', width=200)

    for i in range(len(krit_name)):
        criteria = krit_name[i]
        value = column_values[i] if i < len(column_values) else 0
        criteria_table.insert('', 'end', values=(criteria, value))

    criteria_table.grid(row=0, column=0, columnspan=3)

    criteria_table.bind("<ButtonRelease-1>", on_table_click)

    scrollbar = tk.Scrollbar(criteria_frame, orient='vertical', command=criteria_table.yview)
    scrollbar.grid(row=0, column=3, sticky='ns')

    criteria_table.config(yscrollcommand=scrollbar.set)

    dist_label = tk.Label(root, text="Гипотеза H0")
    dist_label.grid(row=2, column=0)
    values1 = ["Экспоненциальное с масштабом 1.0000 со сдвигом 0.0000", "Логарифмически(ln) Нормальное с масштабом 1.0000 со сдвигом 0.0000 с масштабом 1.0000 со сдвигом 0.0000", "Вейбулла (0.8000) с масштабом 1.0000 со сдвигом 0.0000", "Вейбулла (1.2000) с масштабом 1.0000 со сдвигом 0.0000"]
    distribution_combobox = ttk.Combobox(root, values=values1, state="readonly", width=50)
    distribution_combobox.grid(row=2, column=1, sticky=tk.W)
    distribution_combobox.current(0)
    samples_label = tk.Label(root, text="Количество выборок")
    samples_label.grid(row=3, column=0)
    samples_spinbox = tk.Spinbox(root, from_=0, to=100000, textvariable=tk.IntVar(value=16600))
    samples_spinbox.grid(row=3, column=1, sticky=tk.W)
    
    samples_label1 = tk.Label(root, text="Размер выборок")
    samples_label1.grid(row=4, column=0)
    samples_spinbox1 = tk.Spinbox(root, from_=0, to=100000, textvariable=tk.IntVar(value=100))
    samples_spinbox1.grid(row=4, column=1, sticky=tk.W)

    seed_label = tk.Label(root, text="Начальное значение ГСЧ")
    seed_label.grid(row=5, column=0)
    seed_spinbox = tk.Spinbox(root, from_=0, to=1000, textvariable=tk.IntVar(value=100))
    seed_spinbox.grid(row=5, column=1, sticky=tk.W)

    simulate_button = tk.Button(root, text='Моделировать', command=lambda: do_statistic_modeilng(distribution_combobox.get(), samples_spinbox.get(), samples_spinbox1.get(), seed_spinbox.get()))
    simulate_button.grid(row=6, column=1, sticky=tk.W)

    root.protocol("WM_DELETE_WINDOW", on_main_window_close)
    root.mainloop()

def do_statistic_modeilng(name, number_of_s, size_of, number_of_seed):
    global selected_criteria
    
    if(len(selected_criteria) == 0):
        print("Выберите критерии")
    else:
        for i in (selected_criteria):
            np.random.seed(int(number_of_seed))
            result = modeling_criterion(i, name, int(number_of_s), int(size_of))

            if(name == "Экспоненциальное с масштабом 1.0000 со сдвигом 0.0000"):
                name_1 = i + " N=" + number_of_s + " Эксп(0, 1) n=" + size_of + " ГСЧ=" + number_of_seed
            if(name == "Вейбулла (0.8000) с масштабом 1.0000 со сдвигом 0.0000"):
                name_1 = i + " N=" + number_of_s + " Вей(0.8, 1, 0) n=" + size_of + " ГСЧ=" + number_of_seed
            if(name == "Вейбулла (1.2000) с масштабом 1.0000 со сдвигом 0.0000"):
                name_1 = i + " N=" + number_of_s + " Вей(1.2, 1, 0) n=" + size_of + " ГСЧ=" + number_of_seed
            if(name == "Логарифмически(ln) Нормальное с масштабом 1.0000 со сдвигом 0.0000 с масштабом 1.0000 со сдвигом 0.0000"):
                name_1 = i + " N=" + number_of_s + " Логарифмически(ln) N(0, 1) n=" + size_of + " ГСЧ=" + number_of_seed

            file_name = name_1 + ".dat"
            lines = []
            lines.append(file_name)
            lines.append(number_of_s)
            lines.extend(result)
            # Создаем файл для записи (режим 'w' - для записи)
            with open(file_name, 'w') as file:
                # Записываем все строки в файл
                for line in lines:
                    file.write(str(line) + '\n')
            print(file_name)

def load_file(entry):
    filename = filedialog.askopenfilename()
    entry.delete(0, tk.END)
    entry.insert(0, filename)

def create_p_window(window):
    window.withdraw()
    root1 = tk.Toplevel(window)
    root1.title("Вычисление мощности")
    root1.geometry('930x450')
    def on_close():
        root1.destroy()
        window.deiconify()  # Показать главное окно при закрытии окна вычислений

    root1.protocol("WM_DELETE_WINDOW", on_close)  # Обработчик закрытия окна
   
    def create_criteria_frame(parent):
        frame = tk.Frame(parent, padx=10, pady=5)
        frame.grid(row=3, column=0, columnspan=2, sticky="w")
        tk.Label(frame, text="Критерий").grid(row=0, column=0, sticky="w")
        var = tk.StringVar(value="Правосторонний")  # Используем StringVar вместо IntVar
        tk.Radiobutton(frame, text="Правосторонний", variable=var, value="Правосторонний").grid(row=1, column=0, sticky="w")
        tk.Radiobutton(frame, text="Левосторонний", variable=var, value="Левосторонний").grid(row=2, column=0, sticky="w")
        tk.Radiobutton(frame, text="Двусторонний", variable=var, value="Двусторонний").grid(row=3, column=0, sticky="w")
        return var

    button1 = tk.Button(root1, text="Загрузить G(SH0)", command=lambda: load_file(entry1))
    button1.grid(row=0, column=0, sticky="w", padx=10, pady=5)
    entry1 = tk.Entry(root1, width=130)
    entry1.grid(row=0, column=1, columnspan=2, sticky="w", padx=(0, 0))

    button2 = tk.Button(root1, text="Загрузить G(SH1)", command=lambda: load_file(entry2))
    button2.grid(row=1, column=0, sticky="w", padx=10, pady=5)
    entry2 = tk.Entry(root1, width=130)
    entry2.grid(row=1, column=1, columnspan=2, sticky="w", padx=(0, 0))

    tree = ttk.Treeview(root1, columns=('A', '1-B'), show='headings', style='My.Treeview')
    tree.heading('A', text='A')
    tree.heading('1-B', text='1-B')
    for alpha in ["0.15", "0.1", "0.05", "0.025", "0.01"]:
        tree.insert('', 'end', values=(alpha, ""))
    tree.grid(row=2, column=0, columnspan=3)

    criterion_var = create_criteria_frame(root1)

    def on_calculate():
        file1 = entry1.get()
        file2 = entry2.get()
        criterion = criterion_var.get()
        results = find_power_of(file1, file2, criterion)
        for i, result in enumerate(results):
            tree.set(tree.get_children()[i], column='1-B', value=result)

    calculate_button = tk.Button(root1, text="Вычислить", command=on_calculate)
    calculate_button.grid(row=5, column=0, sticky="w", padx=10, pady=5)

    style = ttk.Style()
    style.theme_use('default')
    style.configure('My.Treeview', borderwidth=1, relief='solid')

    root1.mainloop()

def do_table(criterion, h):
    print(criterion, h)
    n = [10, 20, 30, 40, 50, 100, 150, 200, 300]
    
    data_names = ["Экспоненциальное с масштабом 1.0000 со сдвигом 0.0000", "Логарифмически(ln) Нормальное с масштабом 1.0000 со сдвигом 0.0000 с масштабом 1.0000 со сдвигом 0.0000", 
                  "Вейбулла (0.8000) с масштабом 1.0000 со сдвигом 0.0000", "Вейбулла (1.2000) с масштабом 1.0000 со сдвигом 0.0000"]
    #np.random.seed(100)
    if(h == "H1"):
        data_name = data_names[1]
    elif(h == "H2"):
        data_name = data_names[2]
    elif(h == "H3"):
        data_name = data_names[3]
    
    

    
    result = []
    for i in n:
        a = modeling_criterion(criterion, data_name, 16600, i)
        b = modeling_criterion(criterion, data_names[0], 16600, i)
        
        result.append(find_power_of(a, b, criterion_is(criterion), 1))
       
    #print(result)
    #print(123)
    new_result = []
    for i in result:
        c = []
        for k in i:
            c.append(k)
        new_result.append(c)
    new_result = [[result[j][i] for j in range(len(result))] for i in range(len(result[0]))]
    #print(len(new_result), len(new_result[0]))
    data = {
        "n/a": [10, 20, 30, 40, 50, 100, 150, 200, 300],
        "0.15": new_result[0],
        "0.1": new_result[1],
        "0.05": new_result[2],
        "0.025": new_result[3],
        "0.01": new_result[4]
    }
    
    
    # Преобразуем словарь в DataFrame
    df = pd.DataFrame(data)

    # Записываем DataFrame в файл Excel
    df.to_excel("output.xlsx", index=False)
    print("complete")

class AutocompleteEntry(tk.Entry):
    def __init__(self, lista, *args, **kwargs):
        tk.Entry.__init__(self, *args, **kwargs)
        self.lista = lista
        self.var = self["textvariable"]
        if self.var == '':
            self.var = self["textvariable"] = tk.StringVar()
        self.var.trace('w', self.changed)
        self.bind("<Right>", self.selection)
        self.bind("<Up>", self.up)
        self.bind("<Down>", self.down)
        self.lb_up = False

    def changed(self, name, index, mode):
        if self.var.get() == '':
            self.lb.destroy()
            self.lb_up = False
        else:
            words = self.comparison()
            if words:
                if not self.lb_up:
                    self.lb = tk.Listbox(width=self.winfo_width())
                    self.lb.bind("<Double-Button-1>", self.selection)
                    self.lb.bind("<Right>", self.selection)
                    self.lb.place(x=self.winfo_x(), y=self.winfo_y()+self.winfo_height())
                    self.lb_up = True
                self.lb.delete(0, tk.END)
                for w in words:
                    self.lb.insert(tk.END,w)
            else:
                if self.lb_up:
                    self.lb.destroy()
                    self.lb_up = False

    def selection(self, event):
        if self.lb_up:
            self.var.set(self.lb.get(tk.ACTIVE))
            self.lb.destroy()
            self.lb_up = False
            self.icursor(tk.END)

    def up(self, event):
        if self.lb_up:
            if self.lb.curselection() == ():
                index = '0'
            else:
                index = self.lb.curselection()[0]
            if index != '0':
                self.lb.selection_clear(first=index)
                index = str(int(index)-1)
                self.lb.selection_set(first=index)
                self.lb.activate(index)

    def down(self, event):
        if self.lb_up:
            if self.lb.curselection() == ():
                index = '0'
            else:
                index = self.lb.curselection()[0]
            if index != tk.END:
                self.lb.selection_clear(first=index)
                index = str(int(index)+1)
                self.lb.selection_set(first=index)
                self.lb.activate(index)

    def comparison(self):
        return [w for w in self.lista if w.startswith(self.var.get())]

def generate_data(distribution_name, sample_size):
    #np.random.seed(seed)
    if distribution_name == "Экспоненциальное с масштабом 1.0000 со сдвигом 0.0000":
        sample = np.random.exponential(scale=1.0, size=sample_size)
    elif distribution_name == "Логарифмически(ln) Нормальное с масштабом 1.0000 со сдвигом 0.0000 с масштабом 1.0000 со сдвигом 0.0000":
        sample = np.random.lognormal(mean=0.0, sigma=1.0, size=sample_size)
    elif distribution_name == "Вейбулла (0.8000) с масштабом 1.0000 со сдвигом 0.0000":
        shape = 0.8  # shape parameter (a)
        scale = 1.0  # scale parameter (this is just a multiplier in numpy's weibull)
        shift = 0.0  # shift parameter (if necessary)
        sample = np.random.weibull(a=shape, size=sample_size) * scale + shift

    elif distribution_name == "Вейбулла (1.2000) с масштабом 1.0000 со сдвигом 0.0000":
        sample = np.random.weibull(a=1.2, size=sample_size)
    else:
        print("Неизвестное распределение")
        return -1
    return sample

def modeling(criterion_name, data_name, N):
    r= []
    r1 =[]
    for i in range(16600):
        sample = generate_data(data_name, N)
        r.append(t_stats(criterion_name, sample))

    for i in range(16600):
        sample = generate_data("Экспоненциальное с масштабом 1.0000 со сдвигом 0.0000", N)
        r1.append(t_stats(criterion_name, sample))
    return r, r1

def do_table1(criterion, h):
    np.random.seed(100)
    print(criterion, h)
    n = [10, 20, 30, 40, 50, 100, 150, 200, 300]
    
    data_names = ["Экспоненциальное с масштабом 1.0000 со сдвигом 0.0000", "Логарифмически(ln) Нормальное с масштабом 1.0000 со сдвигом 0.0000 с масштабом 1.0000 со сдвигом 0.0000", 
                  "Вейбулла (0.8000) с масштабом 1.0000 со сдвигом 0.0000", "Вейбулла (1.2000) с масштабом 1.0000 со сдвигом 0.0000"]
    
    if(h == "H1"):
        data_name = data_names[1]
    elif(h == "H2"):
        data_name = data_names[2]
    elif(h == "H3"):
        data_name = data_names[3]


    result = []
    for i in n:
        start_time = time.time()
        r, r1 = modeling(criterion, data_name, i)
        result.append(find_power_of(r1, r, criterion_is(criterion), 1))
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(i, elapsed_time)
    
  
    # Преобразуем словарь в DataFrame
    df = pd.DataFrame(result)
    df_transposed = df.transpose()
    if(criterion == "Крит.показ-ти Шермана/Пиэтра"):
        file_name = "Крит.показ-ти Шермана" + "_" + h + ".xlsx"
    else:
    # Записываем DataFrame в файл Excel
        file_name = criterion + "_" + h + ".xlsx"
    df_transposed.to_excel(file_name, index=False)
    print("complete")

def create_main_window():
    global krit_name
    window = Tk()  
    window.title("Проверка на экспоненциальность")  
    window.geometry('950x250')  

    # Создание основного меню
    menu = Menu(window)
    window.config(menu=menu)

    # Заполнение основного меню
    menu.add_command(label='Моделирование выборки', command=lambda: create_modeling_window(window))
    menu.add_command(label='Проверка на показательность', command=lambda: create_exp_window(window))
    menu.add_command(label='Моделирование статистик критериев', command=lambda: create_statistic_window(window))
    menu.add_command(label='Вычисление мощности', command=lambda: create_p_window(window))
   
      # Замените этот список своим списком слов

    label1 = tk.Label(window, text="Выберите критерий:")
    label1.grid(row=0, column=0)
    entry1 = AutocompleteEntry(krit_name, window, width = 100)
    entry1.grid(row=0, column=1)
    list = ["H1", "H2", "H3"]
    
    label2 = tk.Label(window, text="Относительно гипотезы:")
    label2.grid(row=1, column=0)
    combo_box = ttk.Combobox(window, values=list)
    combo_box.grid(row=1, column=1, sticky='w')

    # Добавляем кнопку
    button = tk.Button(window, text="Оценить мощность", command=lambda: do_table1(entry1.get(), combo_box.get()))
    button.grid(row=2, columnspan=2, sticky='w')
    window.mainloop()

create_main_window()


def do_all_tables():
    n = [10, 20, 30, 40, 50, 100, 150, 200, 300]
    
    data_names = ["Экспоненциальное с масштабом 1.0000 со сдвигом 0.0000", "Логарифмически(ln) Нормальное с масштабом 1.0000 со сдвигом 0.0000 с масштабом 1.0000 со сдвигом 0.0000", 
                  "Вейбулла (0.8000) с масштабом 1.0000 со сдвигом 0.0000", "Вейбулла (1.2000) с масштабом 1.0000 со сдвигом 0.0000"]
    np.random.seed(100)
    for criterion in krit_name: 
        for m in range(1, 4):  
            start_time = time.time() 
            result = []
            for i in n:
                r, r1 = modeling(criterion, data_names[m], i)
                result.append(find_power_of(r1, r, criterion_is(criterion), 1))
            #print(result)
            #print(123)
            new_result = []
            for i in result:
                c = []
                for k in i:
                    c.append(k)
                new_result.append(c)
            new_result = [[result[j][i] for j in range(len(result))] for i in range(len(result[0]))]
            #print(len(new_result), len(new_result[0]))
            data = {
                "n/a": [10, 20, 30, 40, 50, 100, 150, 200, 300],
                "0.15": new_result[0],
                "0.1": new_result[1],
                "0.05": new_result[2],
                "0.025": new_result[3],
                "0.01": new_result[4]
            }
            
            # Преобразуем словарь в DataFrame
            df = pd.DataFrame(data)

            # Записываем DataFrame в файл Excel
            end_time = time.time()
            elapsed_time = end_time - start_time
            file_name = criterion + "_H" + str(m) + ".xlsx"
            print(elapsed_time, file_name)
            df.to_excel(file_name, index=False)
            #print("complete")

