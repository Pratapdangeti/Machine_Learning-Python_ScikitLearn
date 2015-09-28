__author__ = 'pratapdangeti'



from sklearn.linear_model import LinearRegression

x=[[6,2],[8,1],[10,0],[14,2],[18,0]]
y=[[7],[9],[13],[17.5],[18]]

model=LinearRegression()
model.fit(x,y)

x_test = [[8,2],[9,0],[11,2],[16,2],[12,0]]
y_test = [[11],[8.5],[15],[18],[11]]

predictions = model.predict(x_test)

for i, prediction in enumerate(predictions):
    print ('Predicted: %s, Target: %s'% (prediction,y_test[i]))

print ('R-squared: %.2f'%model.score(x_test,y_test))

print (model.intercept_,model.coef_)


