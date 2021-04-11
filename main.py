import model

obj = model.ContenEngine()

FinalResult = obj.predict_final_products('george').sort_values(by = 'percentage', ascending=False)[0:5]

print(FinalResult)
print("Columns list")
print(FinalResult.name)