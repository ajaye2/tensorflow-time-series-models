from tensorflow.keras.layers import Layer, Dense

###
#Gated Linear Unit
###

class GLU(Layer):
  def __init__( self, d_model):
    super(GLU, self).__init__()
    self.linear = Dense(d_model)
    self.sigmoid = Dense(d_model, activation="sigmoid")

  def call(self, inputs):
    return self.linear(inputs) * self.sigmoid(inputs)


# https://keras.io/examples/structured_data/classification_with_grn_and_vsn/