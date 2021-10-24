import random

##tasks = [
##    "Find an expensive italian restaurant in the south. When the system asks for extra preferences, you will specify long stay. When the system suggests something, you will ask for an alternative.", 
##    "Find an expensive italian restaurant in the south. When the system asks for extra preferences, you will specify long stay. When the system suggests something, you will not ask for an alternative.",
####•	Find an expensive chinese restaurant in the south. When the system asks for extra preferences, you will specify good food. When the system suggests something, you will not ask for an alternative. 
####•	Find an expensive chinese restaurant in the south. When the system asks for extra preferences, you will specify good food. When the system suggests something, you will ask for an alternative. 
####•	Find an expensive chinese restaurant in the south. When the system asks for extra preferences, you will specify that it’s not romantic. When the system suggests something, you will not ask for an alternative. 
####•	Find an expensive chinese restaurant in the south. When the system asks for extra preferences, you will specify that it’s not romantic. When the system suggests something, you will ask for an alternative. 
####•	Find an cheap chinese restaurant in the south. When the system asks for extra preferences, you will specify that it’s long stay. When the system suggests something, you will ask for an alternative. 
####•	Find an cheap chinese restaurant in the south. When the system asks for extra preferences, you will not add any. When the system suggests something, you will ask for an alternative. 
##    "Find an expensive european restaurant. The area doesn’t matter. When the system asks for extra preferences, you will specify long stay . When the system suggests something, you will ask for an alternative.",
##    "Find an expensive european restaurant. The area doesn’t matter. When the system asks for extra preferences, you will specify long stay . When the system suggests something, you will not ask for an alternative.",
##    "Find an expensive european restaurant. The area doesn’t matter. When the system asks for extra preferences, you will not  add any . When the system suggests something, you will not ask for an alternative."
##
##]

tasks = [
    "Find an expensive italian restaurant in the south. When the system asks for extra preferences, you will specify long stay. When the system suggests a restaurant, accept the suggestion and don't ask for alternatives.", 
##    "Find an expensive italian restaurant in the south. When the system asks for extra preferences, you will specify long stay. When the system suggests a restaurant, accept the suggestion and don't ask for alternatives.",
##•	Find an expensive chinese restaurant in the south. When the system asks for extra preferences, you will specify good food. When the system suggests something, you will not ask for an alternative. 
##•	Find an expensive chinese restaurant in the south. When the system asks for extra preferences, you will specify good food. When the system suggests something, you will ask for an alternative. 
##•	Find an expensive chinese restaurant in the south. When the system asks for extra preferences, you will specify that it’s not romantic. When the system suggests something, you will not ask for an alternative. 
##•	Find an expensive chinese restaurant in the south. When the system asks for extra preferences, you will specify that it’s not romantic. When the system suggests something, you will ask for an alternative. 
##•	Find an cheap chinese restaurant in the south. When the system asks for extra preferences, you will specify that it’s long stay. When the system suggests something, you will ask for an alternative. 
##•	Find an cheap chinese restaurant in the south. When the system asks for extra preferences, you will not add any. When the system suggests something, you will ask for an alternative. 
##    "Find an expensive european restaurant. The area doesn’t matter. When the system asks for extra preferences, you will specify long stay. When the system suggests a restaurant, accept the suggestion and don't ask for alternatives.",
    "Find an expensive european restaurant. The area doesn’t matter. When the system asks for extra preferences, you will specify long stay. When the system suggests a restaurant, accept the suggestion and don't ask for alternatives.",
    "Find an expensive european restaurant. The area doesn’t matter. When the system asks for extra preferences, you will not add any. When the system suggests a restaurant, accept the suggestion and don't ask for alternatives.",
##    "Find an expensive steakhouse in the east. When the system asks for extra preferences, you will not add any. When the system suggests a restaurant, accept the suggestion and don't ask for alternatives.",
    "Find a cheap persian restaurant in the north. When the system asks for extra preferences, you will not add any. When the system suggests a restaurant, accept the suggestion and don't ask for alternatives."
]


for i in range(3):
    task = random.choice(tasks)
    print(task)
    tasks.remove(task)
    print()
    input("Press enter twice for the next task")
    input()
