from transformers import pipeline

# Initialize the text classification pipeline with the specified model
pipe = pipeline("text-classification", model="achDev/medicalBert", top_k=None)

# Define the Arabic text
text = "هي حالة خطيرة تصيب الرئة وتتطور بسرعة عند الأشخاص الذين يمرون بحالات صحية حرجة (critically ill)، وتتشخص بشكل رئيسي بتسرب السوائل إلى الرئتين مما يجعل التنفس صعبا أو حتى مستحيلا. يستجيب الجسم لهذه الحالات بتفاعلات التهابية التي تكون مفيدة في العادة لمحاربة الأمراض والمساعدة على التئام الجروح، لكن عند بعض المرضى قد تؤدي هذه التفاعلات إلى حدوث تسرب للسوائل من الأوعية الدموية الصغيرة في الرئتين وملئها للحويصلات الهوائية مما يعيق التنفس وانتقال الأكسجين للدم."

# Perform the text classification
result = pipe(text)


print(result)
