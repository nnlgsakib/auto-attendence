from main import load_student_info

def student_logger(student_info):
    loaded_student = student_info.keys()
    print(loaded_student)


if __name__ == "__main__":
    INFO_DIR = 'students_info/'  
    loaded_student_info = load_student_info(INFO_DIR)
    student_logger(loaded_student_info)
