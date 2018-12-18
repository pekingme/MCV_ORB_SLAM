#ifndef ORBVOCABULARY_H
#define ORBVOCABULARY_H

#include "third_party/DBoW2/DBoW2/FORB.h"
#include "third_party/DBoW2/DBoW2/TemplatedVocabulary.h"

namespace MCVORBSLAM
{
    typedef DBoW2::TemplatedVocabulary<DBoW2::FORB::TDescriptor, DBoW2::FORB> ORBVocabulary;
}

#endif // ORBVOCABULARY_H
