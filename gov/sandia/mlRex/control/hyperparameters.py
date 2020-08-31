class InitialParams:

    def __init__(self, imageDimension, epochs, batchSz, testSz, bands):
        self.image_size = imageDimension
        self.image_width, self.image_height = self.image_size, self.image_size
        self.epochs = epochs
        self.batchSz = batchSz
        self.testSz = testSz
        self.bands = bands
        self.input_shape = (self.image_weight, self.image_height, self.bands)
        self.parameters = {"imageSize": self.image_size, "imageWidth": self.image_width, 'imageHeight': self.image_height,
                            'epochs': self.epochs, 'batchSz': self.batchSz, 'testSz': self.testSz, "bands": self.bands,
                            'inputShape': self.inputShape}

    def getTotalPixels(self):
        return self.image_weight*self.image_weight

    def initialParameters(self):
        return self.parmeters