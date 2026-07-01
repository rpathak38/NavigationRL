//
// Created by jemin on 2/25/20.
//

#ifndef _RAISIM_GYM_ANYMAL_RAISIMGYM_ENV_ANYMAL_ENV_RANDOMHEIGHTMAPGENERATOR_HPP_
#define _RAISIM_GYM_ANYMAL_RAISIMGYM_ENV_ANYMAL_ENV_RANDOMHEIGHTMAPGENERATOR_HPP_

#include "raisim/World.hpp"

namespace raisim {

class RandomHeightMapGenerator {
 public:

  enum class GroundType : int {
    HEIGHT_MAP = 0,
    HEIGHT_MAP_DISCRETE = 1,
    STEPS = 2,
    CLIFF = 3,
    NONE = 4
  };

  RandomHeightMapGenerator() = default;

  void setSeed(int seed) {
    terrain_seed_ = seed;
  }

  raisim::HeightMap* generateTerrain(raisim::World* world,
                                     GroundType groundType,
                                     double curriculumFactor,
                                     std::mt19937& gen,
                                     std::uniform_real_distribution<double>& uniDist) {
    std::vector<double> heightVec;
    heightVec.resize(heightMapSampleSize_*heightMapSampleSize_);
    std::unique_ptr<raisim::TerrainGenerator> genPtr;
    double targetRoughness = 0.3;

    /// cliff configuration
    double x_size, y_size;
    int x_num_block, y_num_block;
    int x_sample, y_sample;

    switch (groundType) {
      case GroundType::NONE:
        heightVec.resize(10*10);
        std::fill(heightVec.begin(), heightVec.end(), 0.0);
        return world->addHeightMap(10, 10, 100, 100, 0., 0., heightVec);
        break;

      case GroundType::HEIGHT_MAP:
        terrainProperties_.frequency = 1.0;
        terrainProperties_.zScale = targetRoughness * curriculumFactor;
        terrainProperties_.xSize = 12.5;
        terrainProperties_.ySize = 12.5;
        terrainProperties_.xSamples = 60;
        terrainProperties_.ySamples = 60;
        terrainProperties_.fractalOctaves = 5;
        terrainProperties_.fractalLacunarity = 1.0 + 4.0 * uniDist(gen);
        terrainProperties_.fractalGain = 0.45;
        terrainProperties_.seed = terrain_seed_++;
        terrainProperties_.stepSize = 0.;
        genPtr = std::make_unique<raisim::TerrainGenerator>(terrainProperties_);
        heightVec = genPtr->generatePerlinFractalTerrain();
        { // wall: 2 cells thick, +1m height at borders
          int wallCells = 2;
          for (int i = 0; i < (int)terrainProperties_.xSamples; i++) {
            for (int j = 0; j < (int)terrainProperties_.ySamples; j++) {
              if (i < wallCells || i >= (int)terrainProperties_.xSamples - wallCells ||
                  j < wallCells || j >= (int)terrainProperties_.ySamples - wallCells) {
                heightVec[i * terrainProperties_.ySamples + j] += 1.0;
              }
            }
          }
        }
        return world->addHeightMap(terrainProperties_.xSamples, terrainProperties_.ySamples, terrainProperties_.xSize, terrainProperties_.ySize, 0., 0., heightVec);
        break;

      case GroundType::HEIGHT_MAP_DISCRETE:
        terrainProperties_.frequency = 0.6;
        terrainProperties_.zScale = targetRoughness * curriculumFactor;
        terrainProperties_.xSize = 12.5;
        terrainProperties_.ySize = 12.5;
        terrainProperties_.xSamples = 60;
        terrainProperties_.ySamples = 60;
        terrainProperties_.fractalOctaves = 3;
        terrainProperties_.fractalLacunarity = 3.0;
        terrainProperties_.fractalGain = 0.45;
        terrainProperties_.seed = terrain_seed_++;
        terrainProperties_.stepSize = 0.06 * curriculumFactor;
        genPtr = std::make_unique<raisim::TerrainGenerator>(terrainProperties_);
        heightVec = genPtr->generatePerlinFractalTerrain();
        { // wall: 2 cells thick, +1m height at borders
          int wallCells = 2;
          for (int i = 0; i < (int)terrainProperties_.xSamples; i++) {
            for (int j = 0; j < (int)terrainProperties_.ySamples; j++) {
              if (i < wallCells || i >= (int)terrainProperties_.xSamples - wallCells ||
                  j < wallCells || j >= (int)terrainProperties_.ySamples - wallCells) {
                heightVec[i * terrainProperties_.ySamples + j] += 1.0;
              }
            }
          }
        }
        return world->addHeightMap(terrainProperties_.xSamples, terrainProperties_.ySamples, terrainProperties_.xSize, terrainProperties_.ySize, 0., 0., heightVec);
        break;

      case GroundType::STEPS:
        x_size = 750;
        x_num_block = 15;
        x_sample = x_size / x_num_block;

        heightVec.resize(x_size*x_size);
        for(int xBlock = 0; xBlock < x_num_block; xBlock++) {
          for(int yBlock = 0; yBlock < x_num_block; yBlock++) {
            double height = 0.08 * (uniDist(gen) - 0.5) * curriculumFactor;
            for(int i=0; i<x_sample; i++) {
              for(int j=0; j<x_sample; j++) {
//                heightVec[480 * (12*xBlock+i) + (12*yBlock+j)] = height + xBlock * 0.05 * curriculumFactor;
                heightVec[x_size * (x_sample*xBlock+i) + (x_sample*yBlock+j)] = height;
              }
            }
          }
        }

        return world->addHeightMap(x_size, x_size, 12.0, 12.0, 0., 0., heightVec);
        break;

      case GroundType::CLIFF:
        heightVec.resize(240*240);
        for(int xBlock = 0; xBlock < 40; xBlock++) {
          for(int yBlock = 0; yBlock < 40; yBlock++) {
            double height = 0.06 * (uniDist(gen) - 0.5) * curriculumFactor;
            for(int i=0; i<6; i++) {
              for(int j=0; j<6; j++) {
                heightVec[240 * (6*xBlock+i) + (6*yBlock+j)] = height + xBlock * 0.05 * curriculumFactor;
              }
            }
          }
        }

        return world->addHeightMap(240, 240, 32.0, 32.0, 0., 0., heightVec);
        break;
    }
    return nullptr;
  }

  double getDefaultHeight() { return defaultHeight_; }

 private:
  raisim::TerrainProperties terrainProperties_;
  int heightMapSampleSize_ = 120;
  int terrain_seed_;
  double defaultHeight_ = 0.;
};

}

#endif //_RAISIM_GYM_ANYMAL_RAISIMGYM_ENV_ANYMAL_ENV_RANDOMHEIGHTMAPGENERATOR_HPP_
