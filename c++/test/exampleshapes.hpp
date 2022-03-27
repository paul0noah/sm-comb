//
//  exampleshapes.hpp
//  shape-matching-dd
//
//  Created by Paul Rötzer on 17.11.21.
//

#ifndef exampleshapes_hpp
#define exampleshapes_hpp

#include <stdio.h>
#include "helper/shape.hpp"

namespace test {

Shape getTestShapeBase();
Shape getTestShapeCplxTriTop();
Shape getHorse1Closed();
Shape getHorse1holes();
Shape getHorse2Closed();
Shape getHorse2holes();
Shape getDavid(int numFaces);

} // namespace test

#endif /* exampleshapes_hpp */
